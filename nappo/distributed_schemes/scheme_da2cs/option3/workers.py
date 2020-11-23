import ray
from collections import defaultdict
from .gu_workers import GUWorkerSet
from .c_workers import CWorkerSet


class Workers:
    """
    Class to containing and handling all scheme workers.

    Parameters
    ----------
    algo_factory : func
        A function that creates an algorithm class.
    storage_factory : func
        A function that create a rollouts storage.
    train_envs_factory : func
        A function to create train environments.
    actor_factory : func
        A function that creates a policy.
    test_envs_factory : func
        A function to create test environments.
    col_worker_remote_config : dict
        Ray resource specs for the remote collection workers.
    num_col_workers : int
        Number of remote workers performing collection operations.
    updater_device : torch.device
        CPU or specific GPU to use for local computation.
    broadcast_interval : int
        After how many central updates, model weights should be broadcasted to
        remote collection workers.
    updater_queue_size : int
        Maximum number of data dicts fitting in the updater queue.
    max_collect_requests_pending : int
        Maximum number of collection tasks simultaneously scheduled to each
        collection worker.
    """
    def __init__(self,
                 num_col_workers,
                 algo_factory,
                 actor_factory,
                 storage_factory,
                 train_envs_factory,
                 broadcast_interval=1,
                 updater_device="cpu",
                 updater_queue_size=100,
                 max_collect_requests_pending=2,
                 test_envs_factory=lambda x, y, c: None,
                 col_worker_remote_config={"num_cpus": 1, "num_gpus": 0.5}):

        col_workers_factory = CWorkerSet.worker_set_factory(
            algo_factory=algo_factory,
            actor_factory=actor_factory,
            num_workers=num_col_workers,
            storage_factory=storage_factory,
            test_envs_factory=test_envs_factory,
            train_envs_factory=train_envs_factory,
            worker_remote_config=col_worker_remote_config)

        self._update_workers = GUWorkerSet(
            num_workers=1,
            broadcast_interval=broadcast_interval,
            collection_workers_factory=col_workers_factory,
            worker_remote_config={"num_cpus": 1, "num_gpus": 0.5},
            max_collector_workers_requests_pending=max_collect_requests_pending,
        )

        self.num_workers = len(self._update_workers.remote_workers())

    def update_workers(self):
        """Return local worker"""
        return self._update_workers