import os
import ray
import torch
from shutil import copy2
from ..utils import ray_get_and_free

class UWorker:
    """
    Update worker. Handles actor_critic updates.

    This worker receives gradients from gradient workers and then updates the
    its actor_critic model. Updated weights are asynchronously sent back
    to gradient workers.

    Parameters
    ----------
    grad_workers : WorkerSet
        Set of workers computing and sending gradients to the UWorker.

    Attributes
    ----------
    num_updates : int
        Number of times the actor_critic model has been updated.
    grad_workers : WorkerSet
        Set of workers computing and sending gradients to the UWorker.
    """

    def __init__(self, grad_workers):

        self.num_updates = 0
        self.grad_workers = grad_workers

        # Check remote workers exist
        if len(self.grad_workers.remote_workers()) == 0:
            raise ValueError("""at least 1 grad worker required""")

    def step(self):
        """
        Takes a logical asynchronous optimization step.

        Returns
        -------
        info : dict
            Summary dict of relevant information about the update process.
        """

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
        info["scheme/metrics/gradient_update_delay"] = self.num_updates - info.pop("ac_update_num")

        # Update local worker weights
        self.grad_workers.local_worker().update_networks(gradients)
        e = self.pending_gradients.pop(future)

        # Update counter
        self.num_updates += 1

        # Update remote worker model version
        weights = ray.put({
            "update": self.num_updates,
            "weights": self.grad_workers.local_worker().get_weights()})
        e.set_weights.remote(weights)

        # Call compute_gradients in remote worker again
        future = e.step.remote()
        self.pending_gradients[future] = e

        return info

    def save_model(self, fname):
        """
        Save current version of actor_critic as a torch loadable checkpoint.

        Parameters
        ----------
        fname : str
            Filename given to the checkpoint.

        Returns
        -------
        save_name : str
            Path to saved file.
        """
        torch.save(self.grad_workers.local_worker().actor_critic.state_dict(), fname + ".tmp")
        os.rename(fname + '.tmp', fname)
        save_name = fname + ".{}".format(self.num_updates)
        copy2(fname, save_name)
        return save_name

    def stop(self):
        """Stop remote workers"""
        for e in self.grad_workers.remote_workers():
            e.terminate_worker.remote()

    def adjust_algo_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of Worker.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        """
        self.grad_workers.local_worker().adjust_algo_parameter(parameter_name, new_parameter_value)
        for e in self.grad_workers.remote_workers():
            e.adjust_algo_parameter.remote(parameter_name, new_parameter_value)