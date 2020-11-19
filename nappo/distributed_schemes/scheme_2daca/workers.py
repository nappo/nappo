from .u_worker import UWorker
from .cg_workers import CGWorkerSet


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
    worker_remote_config : dict
        Ray resource specs for the remote workers.
    num_cg_workers : int
        Number of remote workers performing collection/gradient computation
        operations.
    """
    def __init__(self,
                 num_cg_workers,
                 algo_factory,
                 actor_factory,
                 storage_factory,
                 train_envs_factory,
                 test_envs_factory=lambda x, y, c: None,
                 worker_remote_config={"num_cpus": 1, "num_gpus": 0.5}):

        col_grad_workers = CGWorkerSet(
            num_workers=num_cg_workers,
            algo_factory=algo_factory,
            actor_factory=actor_factory,
            storage_factory=storage_factory,
            test_envs_factory=test_envs_factory,
            train_envs_factory=train_envs_factory,
            worker_remote_config=worker_remote_config)
        self._update_worker = UWorker(grad_workers=col_grad_workers)

    def update_worker(self):
        """Return local worker"""
        return self._update_worker