import os
import ray
import torch
from shutil import copy2
from collections import defaultdict

from .cgu_worker import CGUWorker
from .utils import broadcast_message
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config

class CGUWorkerSet(WS):
    """
    Class to better handle the operations of ensembles of workers.

    Parameters
    ----------
    create_algo_instance : func
        A function that creates an algorithm class.
    create_storage_instance : func
        A function that create a rollouts storage.
    create_train_envs_instance : func
        A function to create train environments.
    create_actor_critic_instance : func
        A function that creates a policy.
    create_test_envs_instance : func
        A function to create test environments.
    worker_remote_config : dict
        Ray resource specs for the remote workers.
    fraction_workers : float
        Minimum fraction of the remote workers to complete the data collection
        task to stop collection phase.
    fraction_samples : float
        Minimum fraction of the target steps allowed in a data collection.
    num_workers : int
        Number of remote workers in the worker set.

    Attributes
    ----------
    worker_class : python class
        Worker class to be instantiated to create Ray remote actors.
    remote_config : dict
        Ray resource specs for the remote workers.
    worker_params : dict
        Keyword arguments of the worker_class.
    num_workers : int
        Number of remote workers in the worker set.
    num_updates : int
        Times step() function has been called.
    fraction_workers : float
        Minimum fraction of the remote workers to complete the data collection
        task to stop collection phase.
    fraction_samples : float
        Minimum fraction of the target steps allowed in a data collection.
    """

    def __init__(self,
                 create_algo_instance,
                 create_storage_instance,
                 create_train_envs_instance,
                 create_actor_critic_instance,
                 create_test_envs_instance=lambda x, y, c: None,
                 worker_remote_config=default_remote_config,
                 fraction_workers=0.8,
                 fraction_samples=0.5,
                 num_workers=1):

        self.worker_class = CGUWorker
        default_remote_config.update(worker_remote_config)
        self.remote_config = default_remote_config
        self.worker_params = {
            "create_algo_instance": create_algo_instance,
            "create_storage_instance": create_storage_instance,
            "create_test_envs_instance": create_test_envs_instance,
            "create_train_envs_instance": create_train_envs_instance,
            "create_actor_critic_instance": create_actor_critic_instance,
        }

        super(CGUWorkerSet, self).__init__(
            worker=self.worker_class,
            worker_params=self.worker_params,
            worker_remote_config=self.remote_config,
            num_workers=num_workers)

        self.num_updates = 0
        self.num_workers = num_workers
        self.fraction_samples = fraction_samples if self.num_workers > 1 else 1.0
        self.fraction_workers = fraction_workers if self.num_workers > 1 else 1.0

        # Check remote workers exist
        if len(self.remote_workers()) == 0:
            raise ValueError("""at least 1 grad worker required""")

        # Setup the distributed processes for gradient averaging
        ip = ray.get(self.remote_workers()[0].get_node_ip.remote())
        port = ray.get(self.remote_workers()[0].find_free_port.remote())
        address = "tcp://{ip}:{port}".format(ip=ip, port=port)
        ray.get([worker.setup_torch_data_parallel.remote(
            address, i, len(self.remote_workers()), "nccl")
                 for i, worker in enumerate(self.remote_workers())])

    def step(self):
        """
        Takes a logical optimization step.

        Returns
        -------
        info : dict
            Summary dict of relevant information about the update process.
        """

        # Start data collection in all workers
        broadcast_message("sample", b"start-continue")

        pending_samples = [e.collect_data.remote(
            min_fraction=self.fraction_samples) for e in self.remote_workers()]

        # Keep checking how many workers have finished until percent% are ready
        samples_ready, samples_not_ready = ray.wait(
            pending_samples, num_returns=len(pending_samples), timeout=0.5)
        while len(samples_ready) < (self.num_workers * self.fraction_workers):
            samples_ready, samples_not_ready = ray.wait(
                pending_samples, num_returns=len(pending_samples), timeout=0.5)

        # Send stop message to the workers
        broadcast_message("sample", b"stop")

        # Compute model updates
        results = ray.get([e.step.remote() for e in self.remote_workers()])

        # Merge worker results
        step_metrics = defaultdict(float)
        for info in results:
            info["scheme/metrics/gradient_update_delay"] = self.num_updates - info.pop("ac_update_num")
            for k, v in info.items(): step_metrics[k] += v

        # Update info dict
        info = {k: v / self.num_workers for k, v in step_metrics.items()}

        # Update counters
        self.num_updates += 1

        return info

    def update_algo_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of Worker.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        """
        for e in self.remote_workers():
            e.update_algo_parameter.remote(parameter_name, new_parameter_value)

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
        model_dict = ray.get(self.remote_workers()[0].get_weights.remote())
        torch.save(model_dict, fname + ".tmp")
        os.rename(fname + '.tmp', fname)
        save_name = fname + ".{}".format(self.num_updates)
        copy2(fname, save_name)
        return save_name