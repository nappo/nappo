import os
import ray
import time
import torch
from shutil import copy2
from collections import defaultdict
from ..utils import ray_get_and_free, average_gradients


class UWorker:
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

    def __init__(self, grad_workers):

        self.num_updates = 0
        self.grad_workers = grad_workers
        self.num_workers = len(self.grad_workers.remote_workers())

        # Check remote workers exist
        if len(self.grad_workers.remote_workers()) == 0:
            raise ValueError("""at least 1 grad worker required""")

    def sync_weights(self):
        """Synchronize gradient worker models with updater worker model"""
        weights = ray.put({
            "update": self.num_updates,
            "weights": self.grad_workers.local_worker().get_weights()})
        for e in self.grad_workers.remote_workers():
            e.set_weights.remote(weights)

    def step(self):
        """
        Takes a logical asynchronous optimization step.

        Returns
        -------
        info : dict
            Summary dict of relevant information about the update process.
        """

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
        self.grad_workers.local_worker().update_networks(average_gradients(to_average))
        avg_grads_t = time.time() - t

        # Update workers with current weights
        t = time.time()
        self.sync_weights()
        sync_grads_t = time.time() - t

        # Update counter
        self.num_updates += 1

        # Update info dict
        info = {k: v/self.num_workers if k != "collected_samples" else
        v for k, v in step_metrics.items()}
        info.update({"scheme/seconds_to/avg_grads": avg_grads_t})
        info.update({"scheme/seconds_to/sync_grads": sync_grads_t})

        return info

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