import os
import ray
import time
import torch
from shutil import copy2
import threading
from six.moves import queue
from collections import defaultdict
from ..base.worker import Worker as W
from ..utils import ray_get_and_free, average_gradients


class UWorker(W):
    """
    Update worker. Handles actor updates.

    This worker receives gradients from gradient workers and then updates the
    its actor model. Updated weights are synchronously sent back
    to gradient workers.

    Parameters
    ----------
    grad_workers : WorkerSet
        Set of workers computing and sending gradients to the UWorker.

    Attributes
    ----------
    num_updates : int
        Number of times the actor model has been updated.
    grad_workers : WorkerSet
        Set of workers computing and sending gradients to the UWorker.
    num_workers : int
        number of remote workers computing gradients.
    """

    def __init__(self,
                 index_worker,
                 grad_workers_factory,
                 grad_execution="decentralised",
                 grad_communication="synchronous",
                 update_execution="centralised",
                 local_device=None):

        super(UWorker, self).__init__(index_worker)

        self.num_updates = 0
        self.grad_execution = grad_execution
        self.update_execution = update_execution
        self.grad_communication = grad_communication

        # Computation device
        dev = local_device or "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        # Create CWorkerSet instance
        add_local_worker = update_execution == "centralised"
        self.grad_workers = grad_workers_factory(local_device, add_local_worker)





        self.local_worker = self.grad_workers.local_worker()
        self.num_workers = len(self.grad_workers.remote_workers())

        # Queue
        self.outqueue = queue.Queue()

        # Create UpdaterThread
        self.updater = UpdaterThread(
            output_queue=self.outqueue,
            grad_workers=self.grad_workers,
            grad_communication=grad_communication,
            grad_execution=grad_execution)


        # Print worker information
        if index_worker > 0: self.print_worker_info()

    def step(self):
        """ _ """

        if self.grad_communication == "synchronous":
            self.updater.step()

        new_info = self.outqueue.get(timeout=300)

        return new_info

    def save_model(self, fname):
        """
        Save current version of actor as a torch loadable checkpoint.

        Parameters
        ----------
        fname : str
            Filename given to the checkpoint.

        Returns
        -------
        save_name : str
            Path to saved file.
        """
        torch.save(self.grad_workers.local_worker().actor.state_dict(), fname + ".tmp")
        os.rename(fname + '.tmp', fname)
        save_name = fname + ".{}".format(self.num_updates)
        copy2(fname, save_name)
        return save_name

    def stop(self):
        """Stop remote workers"""
        for e in self.grad_workers.remote_workers():
            e.terminate_worker.remote()

    def update_algo_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of Worker.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        """
        self.grad_workers.local_worker().update_algo_parameter(parameter_name, new_parameter_value)
        for e in self.grad_workers.remote_workers():
            e.update_algo_parameter.remote(parameter_name, new_parameter_value)


class UpdaterThread(threading.Thread):
    """
    This class receives data from the workers and continuously updates central actor.

    Parameters
    ----------
    input_queue : queue.Queue
        Queue to store the data dicts received and pending to be processed.
    output_queue : queue.Queue
        Queue to store the info dicts resulting from the model update operation.
    local_worker : Worker
        Local worker that acts as a parameter server.

    Attributes
    ----------
    local_worker : Worker
        Local worker that acts as a parameter server.
    input_queue : queue.Queue
        Queue to store the data dicts received and pending to be processed.
    output_queue : queue.Queue
        Queue to store the info dicts resulting from the model update operation.
    stopped : bool
        Whether or not the thread in running.
    """

    def __init__(self,
                 output_queue,
                 grad_workers,
                 grad_execution="distributed",
                 grad_communication="synchronous"):

        threading.Thread.__init__(self)

        self.stopped = False
        self.num_updates = 0
        self.outqueue = output_queue
        self.grad_workers = grad_workers
        self.grad_execution = grad_execution
        self.grad_communication = grad_communication
        self.local_worker = self.grad_workers.local_worker()
        self.remote_workers = self.grad_workers.remote_workers()
        self.num_workers = len(self.grad_workers.remote_workers())

        if grad_execution == "centralised" and grad_communication == "synchronous":
            pass

        elif grad_execution == "centralised" and grad_communication == "asynchronous":
            # Start UpdaterThread
            self.start()

        elif grad_execution == "decentralised" and grad_communication == "synchronous":
            pass

        elif grad_execution == "decentralised" and grad_communication == "asynchronous":
            # Start UpdaterThread
            self.start()

        else:
            raise NotImplementedError

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        """
        Continuously pulls data from the input queue, computes gradients,
        updates the local actor model and places information in the
        output queue.
        """

        if self.grad_execution == "centralised" and self.grad_communication == "synchronous":
            _, info = self.local_worker.step()
            self.local_worker.apply_gradients()

        elif self.grad_execution == "centralised" and self.grad_communication == "asynchronous":
            _, info = self.local_worker.step()
            self.local_worker.apply_gradients()

        elif self.grad_execution == "decentralised" and self.grad_communication == "synchronous":

            to_average = []
            pending_gradients = {}
            step_metrics = defaultdict(float)

            # Call for gradients from all workers
            for e in self.grad_workers.remote_workers():
                future = e.step.remote()
                pending_gradients[future] = e

            # Wait for workers to send back gradients
            while pending_gradients:

                # Get gradients 1 by 1
                wait_results = ray.wait(list(pending_gradients.keys()))
                ready_list = wait_results[0]
                future = ready_list[0]
                gradients, info = ray_get_and_free(future)
                pending_gradients.pop(future)

                # Update info dict
                info["scheme/metrics/gradient_update_delay"] = self.num_updates - info.pop("ac_update_num")

                # Update counters
                for k, v in info.items(): step_metrics[k] += v

                # Store gradients to average later
                to_average.append(gradients)

            # Average and apply gradients
            t = time.time()
            self.grad_workers.local_worker().update_networks(
                average_gradients(to_average))
            avg_grads_t = time.time() - t

            # Update workers with current weights
            t = time.time()
            self.sync_weights()
            sync_grads_t = time.time() - t

            # Update info dict
            info = {k: v / self.num_workers if k != "collected_samples" else
            v for k, v in step_metrics.items()}
            info.update({"scheme/seconds_to/avg_grads": avg_grads_t})
            info.update({"scheme/seconds_to/sync_grads": sync_grads_t})

        elif self.grad_execution == "decentralised" and self.grad_communication == "asynchronous":
            # If first call, call for gradients from all workers
            if self.num_updates == 0:
                self.pending_gradients = {}
                for e in self.grad_workers.remote_workers():
                    future = e.step.remote()
                    self.pending_gradients[future] = e

            # Wait for first gradients ready
            wait_results = ray.wait(list(self.pending_gradients.keys()))
            ready_list = wait_results[0]
            future = ready_list[0]

            # Get gradients
            gradients, info = ray_get_and_free(future)

            # Update info dict
            info["scheme/metrics/gradient_update_delay"] = self.num_updates - info.pop(
                "ac_update_num")

            # Update local worker weights
            self.grad_workers.local_worker().update_networks(gradients)
            e = self.pending_gradients.pop(future)

            # Update remote worker model version
            weights = ray.put({
                "update": self.num_updates,
                "weights": self.local_worker.get_weights()})
            e.set_weights.remote(weights)

            # Call compute_gradients in remote worker again
            future = e.step.remote()
            self.pending_gradients[future] = e

        else:
            raise NotImplementedError

        # Update counter
        self.num_updates += 1

        # Add step info to queue
        self.outqueue.put(info)

    def sync_weights(self):
        """Synchronize gradient worker models with updater worker model"""
        weights = ray.put({
            "update": self.num_updates,
            "weights": self.local_worker.get_weights()})
        for e in self.remote_workers: e.set_weights.remote(weights)

    def stop(self):
        """Stop updating the local policy."""
        self.stopped = True