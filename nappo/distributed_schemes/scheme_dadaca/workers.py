from .c_workers import CWorkerSet
from .g_workers import GWorkerSet
from .u_worker import UWorker


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
    num_col_workers : int
        Number of remote workers performing collection operations.
    col_worker_remote_config : dict
        Ray resource specs for the remote collection workers.
    num_grad_workers : int
        Number of remote workers performing collection operations.
    grad_worker_remote_config : dict
        Ray resource specs for the remote gradient computation workers.
    """
    def __init__(self,
                 num_col_workers,
                 num_grad_workers,
                 algo_factory,
                 actor_factory,
                 storage_factory,
                 train_envs_factory,
                 test_envs_factory=lambda x, y, c: None,
                 col_worker_remote_config={"num_cpus": 1, "num_gpus": 0.5},
                 grad_worker_remote_config={"num_cpus": 1, "num_gpus": 0.5}):

        col_workers_factory = CWorkerSet.worker_set_factory(
            num_workers=num_col_workers//num_grad_workers,
            test_envs_factory=test_envs_factory,
            train_envs_factory=train_envs_factory,
            actor_factory=actor_factory,
            worker_remote_config=col_worker_remote_config)
        grad_workers = GWorkerSet(
            algo_factory=algo_factory,
            storage_factory=storage_factory,
            actor_factory=actor_factory,
            collection_workers_factory=col_workers_factory,
            num_workers=num_grad_workers,
            worker_remote_config=grad_worker_remote_config)
        self._update_worker = UWorker(grad_workers)

    def update_workers(self):
        """Return local worker"""
        return self._update_worker