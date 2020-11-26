import os
import ray
import time
import torch
import threading
from shutil import copy2
from copy import deepcopy
from six.moves import queue
from functools import partial
from collections import defaultdict, deque

from ..base.worker import Worker as W
from ..utils import ray_get_and_free, broadcast_message


class GWorker(W):
    """
    Worker class handling gradient computation.

    This class wraps an actor instance, a storage class instance and a
    worker set of remote data collection workers. It receives data from the
    collection workers and computes gradients following a logic defined in
    function self.step(), which will be called from the Learner class.

    Parameters
    ----------
    index_worker : int
        Worker index.
    algo_factory : func
        A function that creates an algorithm class.
    storage_factory : func
        A function that create a rollouts storage.
    actor_factory : func
        A function that creates a policy.
    collection_workers_factory : func
        A function that creates a sets of data collection workers.
    initial_weights : ray object ID
        Initial model weights.
    max_collect_requests_pending : int
        maximum number of collection tasks simultaneously scheduled to each
        collection worker.

    Attributes
    ----------
    index_worker : int
        Index assigned to this worker.
    actor : nn.Module
        An actor class instance.
    algo : an algorithm class
        An algorithm class instance.
    storage : a rollout storage class
        A Storage class instance.
    ac_version : int
        Number of times the current actor version been has been updated.
    latest_weights : ray object ID
        Last received model weights.
    """

    def __init__(self,
                 index_worker,
                 col_workers_factory,
                 col_communication="synchronous",
                 col_execution="distributed",
                 initial_weights=None,
                 device=None):

        super(GWorker, self).__init__(index_worker)

        # Define counters and other attributes
        self.iter = 0
        self.communication = col_communication

        # Computation device
        dev = device or "cuda" if torch.cuda.is_available() else "cpu"

        # Create CWorkerSet instance
        self.c_workers = col_workers_factory(dev, initial_weights)
        self.local_worker = self.c_workers.local_worker()
        self.remote_workers = self.c_workers.remote_workers()

        # Get Actor Critic instance
        self.actor = self.local_worker.actor

        # Get Algorithm instance
        self.algo = self.local_worker.algo

        # Get storage instance
        self.storage = deepcopy(self.local_worker.storage)

        # Queue
        self.inqueue = queue.Queue(maxsize=100)

        # Create CollectorThread
        self.collector = CollectorThread(
            input_queue=self.inqueue,
            local_worker=self.local_worker,
            remote_workers=self.remote_workers,
            col_communication=col_communication,
            col_execution=col_execution,
            broadcast_interval=1)

        # Print worker information
        self.print_worker_info()

    @property
    def actor_version(self):
        return self.local_worker.actor_version

    def step(self, fraction_workers=1.0, fraction_samples=1.0, distribute_gradients=False):
        """
        Perform logical learning step. Training proceeds receiving data samples
        from collection workers and computations policy gradients.

        Returns
        -------
        grads: list of tensors
            List of actor gradients.
        info : dict
            Summary dict of relevant step information.
        """

        # Collect data and prepare data batches
        if self.iter % (self.algo.num_epochs * self.algo.num_mini_batch) == 0:

            if self.communication == "synchronous":
                self.collector.step(fraction_workers, fraction_samples)

            data, self.col_info = self.inqueue.get(timeout=300)
            self.storage.add_data(data)
            self.storage.before_update(self.actor, self.algo)
            self.batches = self.storage.generate_batches(
                self.algo.num_mini_batch, self.algo.mini_batch_size,
                self.algo.num_epochs, self.actor.is_recurrent)

        # Compute gradients, get algo info
        grads, info = self.compute_gradients(self.batches.__next__())

        # Add extra information to info dict
        info.update(self.col_info)
        self.col_info.update({"collected_samples": 0})
        info.update({"grad_version": self.local_worker.actor_version})

        if distribute_gradients:
            grads = None
            self.distribute_gradients(grads)

        self.iter += 1

        return grads, info

    def compute_gradients(self, batch):
        """
        Calculate actor gradients and update networks.

        Parameters
        ----------
        batch : dict
            data batch containing all required tensors to compute algo loss.

        Returns
        -------
        grads: list of tensors
            List of actor gradients.
        info : dict
            Summary dict with relevant gradient-related information.
        """

        t = time.time()
        grads, info = self.algo.compute_gradients(batch)
        info.update({"time/compute_grads": time.time() - t})

        return grads, info

    def distribute_gradients(self, grads):
        """ _ """
        t = time.time()
        if torch.cuda.is_available():
            for g in grads:
                torch.distributed.all_reduce(g, op=torch.distributed.ReduceOp.SUM)
        else:
            torch.distributed.all_reduce_coalesced(grads, op=torch.distributed.ReduceOp.SUM)

        for p in self.actor.parameters():
            if p.grad is not None:
                p.grad /= self.distributed_world_size
        avg_grads_t = time.time() - t
        return avg_grads_t

    def apply_gradients(self, gradients=None):
        """Update Actor Critic model"""
        self.local_worker.actor_version += 1
        self.algo.apply_gradients(gradients)
        if self.communication == "synchronous":
            self.collector.broadcast_new_weights()

    def set_weights(self, weights):
        """
        Update the worker actor version with provided weights.

        weights: dict of tensors
            Dict containing actor weights to be set.
        """
        self.local_worker.actor_version = weights["version"]
        self.local_worker.algo.set_weights(weights["weights"])

    def update_algo_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of Worker.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        """
        self.local_worker.update_algo_parameter(parameter_name, new_parameter_value)
        for e in self.remote_workers:
            e.update_algo_parameter.remote(parameter_name, new_parameter_value)


        self.algo.update_algo_parameter(parameter_name, new_parameter_value)
        for e in self.c_workers.remote_workers():
            e.update_algo_parameter.remote(parameter_name, new_parameter_value)

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
        torch.save(self.local_worker.actor.state_dict(), fname + ".tmp")
        os.rename(fname + '.tmp', fname)
        save_name = fname + ".{}".format(self.local_worker.actor_version)
        copy2(fname, save_name)
        return save_name

    def stop(self):
        """Stop collecting data."""
        self.collector.stopped = True
        for e in self.collector.remote_workers:
            e.terminate_worker.remote()


class CollectorThread(threading.Thread):
    """
    This class receives data from the workers and queues it to the updater queue.


    Parameters
    ----------
    inqueue : queue.Queue
        Queue to store the data dicts received and pending to be processed.
    local_worker : Worker
        Local worker that acts as a parameter server.
    remote_workers : list of Workers
        Set of workers collecting and sending rollouts.
    broadcast_interval : int
        After how many central updates, model weights should be broadcasted to
        remote collection workers.
    max_collect_requests_pending : int
        Maximum number of collection tasks simultaneously scheduled to each
        collection worker.

    Attributes
    ----------
    input_queue : queue.Queue
        Queue to store the data dicts received and pending to be processed.
    local_worker : Worker
        Local worker that acts as a parameter server.
    remote_workers : list of Workers
        Set of workers collecting and sending rollouts.
    broadcast_interval : int
        After how many central updates, model weights should be broadcasted to
        remote collection workers.
    num_sent_since_broadcast : int
        Number of data dicts received since last model weights were broadcasted.
    num_workers : int
        number of remote workers computing gradients.
    collector_tasks : TaskPool
        Task pool to track remote workers in-flight collection tasks.
    stopped : bool
        Whether or not the thread in running.
    """

    def __init__(self,
                 input_queue,
                 local_worker,
                 remote_workers,
                 col_communication="synchronous",
                 col_execution="distributed",
                 broadcast_interval=1):

        threading.Thread.__init__(self)

        self.stopped = False
        self.inqueue = input_queue
        self.col_execution = col_execution
        self.col_communication = col_communication
        self.broadcast_interval = broadcast_interval

        self.local_worker = local_worker
        self.remote_workers = remote_workers
        self.num_workers = len(self.remote_workers)

        # Counters and metrics
        self.num_sent_since_broadcast = 0
        self.metrics = defaultdict(partial(deque, maxlen=100))

        if col_execution == "centralised" and col_communication == "synchronous":
            pass

        elif col_execution == "centralised" and col_communication == "asynchronous":
            # Start CollectorThread
            self.start()

        elif col_execution == "decentralised" and col_communication == "synchronous":
            pass

        elif col_execution == "decentralised" and col_communication == "asynchronous":
            # Start CollectorThread
            self.start()
            self.pending_tasks = {}
            self.broadcast_new_weights()
            for w in self.remote_workers:
                for _ in range(2):
                    future = w.collect_data.remote()
                    self.pending_tasks[future] = w

        else:
            raise NotImplementedError


    def run(self):
        while not self.stopped:

            # First, collect data
            self.step()

            # Then, update counter and broadcast weights to worker if necessary
            self.num_sent_since_broadcast += 1
            if self.should_broadcast():
                self.broadcast_new_weights()

    def step(self, fraction_workers=1.0, fraction_samples=1.0):
        """
        Continuously collects data from remote workers and puts it
        in the updater queue.
        """

        if self.col_execution == "centralised" and self.col_communication == "synchronous":

            rollouts = self.local_worker.collect_data(min_fraction=fraction_samples)
            self.inqueue.put(rollouts)

        elif self.col_execution == "centralised" and self.col_communication == "asynchronous":
            rollouts = self.local_worker.collect_data(min_fraction=fraction_samples)
            self.inqueue.put(rollouts)

        elif self.col_execution == "decentralised" and self.col_communication == "synchronous":

            fraction_samples = fraction_samples if self.num_workers > 1 else 1.0
            fraction_workers = fraction_workers if self.num_workers > 1 else 1.0

            # Start data collection in all workers
            broadcast_message("sample", b"start-continue")

            pending_samples = [e.collect_data.remote(
                min_fraction=fraction_samples) for e in self.remote_workers]

            # Keep checking how many workers have finished until percent% are ready
            samples_ready, samples_not_ready = ray.wait(pending_samples,
                num_returns=len(pending_samples), timeout=0.5)
            while len(samples_ready) < (self.num_workers * fraction_workers):
                samples_ready, samples_not_ready = ray.wait(pending_samples,
                    num_returns=len(pending_samples), timeout=0.5)

            # Send stop message to the workers
            broadcast_message("sample", b"stop")

            # Compute model updates
            for r in pending_samples: self.inqueue.put(ray_get_and_free(r))

        elif self.col_execution == "decentralised" and self.col_communication == "asynchronous":

            # Wait for first worker to finish
            wait_results = ray.wait(list(self.pending_tasks.keys()))
            future = wait_results[0][0]
            w = self.pending_tasks.pop(future)

            # Retrieve rollouts and add them to queue
            self.inqueue.put(ray_get_and_free(future))

            # Then, update counter and broadcast weights to worker if necessary
            self.num_sent_since_broadcast += 1
            if self.should_broadcast():
                self.broadcast_new_weights()

            # Schedule a new collection task
            future = w.collect_data.remote(min_fraction=fraction_samples)
            self.pending_tasks[future] = w

        else:
            raise NotImplementedError

    def should_broadcast(self):
        """Returns whether broadcast() should be called to update weights."""
        return self.num_sent_since_broadcast >= self.broadcast_interval

    def broadcast_new_weights(self):
        """Broadcast a new set of weights from the local worker."""
        if self.num_workers > 0:
            latest_weights = ray.put({
                "version": self.local_worker.actor_version,
                "weights": self.local_worker.get_weights()})
            for e in self.remote_workers:
                e.set_weights.remote(latest_weights)
            self.num_sent_since_broadcast = 0


