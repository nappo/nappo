from nappo.schemes.workers_dadacs import CWorkerSet, GWorkerSet, UWorker

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
                 create_algo_instance,
                 create_storage_instance,
                 create_test_envs_instance,
                 create_train_envs_instance,
                 create_actor_critic_instance,
                 col_worker_remote_config={"num_cpus": 1, "num_gpus": 0.5},
                 grad_worker_remote_config={"num_cpus": 1, "num_gpus": 0.5}):

        col_workers = CWorkerSet.worker_set_factory(
            num_workers=num_col_workers//num_grad_workers,
            create_test_envs_instance=create_test_envs_instance,
            create_train_envs_instance=create_train_envs_instance,
            create_actor_critic_instance=create_actor_critic_instance,
            worker_remote_config=col_worker_remote_config)
        grad_workers = GWorkerSet(
            create_algo_instance=create_algo_instance,
            create_storage_instance=create_storage_instance,
            create_actor_critic_instance=create_actor_critic_instance,
            create_collection_worker_set_instance=col_workers,
            num_workers=num_grad_workers,
            worker_remote_config=grad_worker_remote_config)
        self._update_worker = UWorker(grad_workers)

    def update_worker(self):
        """Return local worker"""
        return self._update_worker