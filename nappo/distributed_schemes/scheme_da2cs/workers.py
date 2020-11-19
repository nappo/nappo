from .gu_worker2 import GUWorker
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
                 updater_queue_size=100,
                 updater_device="cuda:0",
                 max_collect_requests_pending=2,
                 test_envs_factory=lambda x, y, c: None,
                 col_worker_remote_config={"num_cpus": 1, "num_gpus": 0.5}):

        col_workers = CWorkerSet(
            num_workers=num_col_workers,
            algo_factory=algo_factory,
            storage_factory=storage_factory,
            test_envs_factory=test_envs_factory,
            train_envs_factory=train_envs_factory,
            actor_factory=actor_factory,
            worker_remote_config=col_worker_remote_config)
        self._update_worker = GUWorker(
            col_workers,
            device=updater_device,
            broadcast_interval=broadcast_interval,
            updater_queue_size=updater_queue_size,
            max_collect_requests_pending=max_collect_requests_pending)

    def update_workers(self):
        """Return local worker"""
        return self._update_worker