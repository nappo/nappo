import ray
from .cg_worker import CGWorker
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config


class CGWorkerSet(WS):
    """
    Class to better handle the operations of ensembles of CGWorkers.

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
    """

    def __init__(self,
                 create_algo_instance,
                 create_storage_instance,
                 create_train_envs_instance,
                 create_actor_critic_instance,
                 create_test_envs_instance=lambda x, y, c: None,
                 worker_remote_config=default_remote_config,
                 num_workers=1):

        self.worker_class = CGWorker
        default_remote_config.update(worker_remote_config)
        self.remote_config = default_remote_config
        self.worker_params = {
            "create_algo_instance": create_algo_instance,
            "create_storage_instance": create_storage_instance,
            "create_test_envs_instance": create_test_envs_instance,
            "create_train_envs_instance": create_train_envs_instance,
            "create_actor_critic_instance": create_actor_critic_instance,
        }

        super(CGWorkerSet, self).__init__(
            worker=self.worker_class,
            worker_params=self.worker_params,
            worker_remote_config=self.remote_config,
            num_workers=num_workers)


