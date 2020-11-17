from .u_worker import UWorker
from .cg_workers import CGWorkerSet


class Workers:
    """
    Class to containing and handling all scheme workers.

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
    num_cg_workers : int
        Number of remote workers performing collection/gradient computation
        operations.
    """
    def __init__(self,
                 num_cg_workers,
                 create_algo_instance,
                 create_storage_instance,
                 create_test_envs_instance,
                 create_train_envs_instance,
                 create_actor_critic_instance,
                 worker_remote_config={"num_cpus": 1, "num_gpus": 0.5}):

        col_grad_workers = CGWorkerSet(
            num_workers=num_cg_workers,
            create_algo_instance=create_algo_instance,
            create_storage_instance=create_storage_instance,
            create_test_envs_instance=create_test_envs_instance,
            create_train_envs_instance=create_train_envs_instance,
            create_actor_critic_instance=create_actor_critic_instance,
            worker_remote_config=worker_remote_config)
        self._update_worker = UWorker(grad_workers=col_grad_workers)

    def update_worker(self):
        """Return local worker"""
        return self._update_worker