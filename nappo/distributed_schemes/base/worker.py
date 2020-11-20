import os
import ray
import torch
import logging
from shutil import copy2
from ray.services import get_node_ip_address
from .utils import find_free_port

logger = logging.getLogger(__name__)

default_remote_config = {
    "num_cpus": 1,
    "num_gpus": 0.5,
}


class Worker:
    """
    Class containing common worker functionality.

    Parameters
    ----------
    index_worker : int
        Worker index.

    Attributes
    ----------
    index_worker : int
        Index assigned to this worker.
    actor : nn.Module
        An actor class instance.
    """

    def __init__(self, index_worker):
        self.index_worker = index_worker
        self.actor = None # Initialize in inherited class

    @classmethod
    def as_remote(cls,
                  num_cpus=None,
                  num_gpus=None,
                  memory=None,
                  object_store_memory=None,
                  resources=None):
        """
        Creates a Worker instance as a remote ray actor.

        Parameters
        ----------
        num_cpus : int
            The quantity of CPU cores to reserve for this Worker class.
        num_gpus  : float
            The quantity of GPUs to reserve for this Worker class.
        memory : int
            The heap memory quota for this actor (in bytes).
        object_store_memory : int
            The object store memory quota for this actor (in bytes).
        resources: Dict[str, float]
            The default resources required by the actor creation task.

        Returns
        -------
        W : Worker
            A ray remote actor Worker class.
        """
        W = ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources)(cls)
        return W

    def print_worker_info(self):
        """Print information about this worker, including index and resources assigned"""
        s = "Created {} with worker_index {}".format(
            str(type(self).__name__), self.index_worker)
        if self.index_worker != 0:
            s += ", in machine {} using gpus {}".format(
                ray._private.services.get_node_ip_address(),
                ray.get_gpu_ids())
        logger.warning(s)

    def get_weights(self):
        """Returns current actor.state_dict() weights"""
        return {k: v.cpu() for k, v in self.actor.state_dict().items()}

    def save_model(self, fname, iter):
        """
        Save current version of actor as a torch loadable checkpoint.

        Parameters
        ----------
        fname : str
            Filename given to the checkpoint.
        iter: int
            Training iteration.

        Returns
        -------
        save_name : str
            Path to saved file.
        """
        torch.save(self.actor.state_dict(), fname + ".tmp")
        os.rename(fname + '.tmp', fname)
        save_name = fname + ".{}".format(iter)
        copy2(fname, save_name)
        return save_name

    def terminate_worker(self):
        """Terminate this ray actor"""
        ray.actor.exit_actor()

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return get_node_ip_address()

    def find_free_port(self):
        """Returns a free port on the current node."""
        return find_free_port()

    def setup_torch_data_parallel(self, url, world_rank, world_size, backend):
        """Join a torch process group for distributed SGD."""
        torch.distributed.init_process_group(
            backend=backend,
            init_method=url,
            rank=world_rank,
            world_size=world_size)
        self.distributed_world_size = world_size

    @staticmethod
    def get_host():
        """Return node name where this Worker is being executed."""
        return os.uname()[1]
