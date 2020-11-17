from .c_worker import CWorker
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config

class CWorkerSet(WS):
    """
    Class to better handle the operations of ensembles of workers.

    Parameters
    ----------
    initial_weights : ray object ID
        Initial model weights.
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
        Keyword parameters of the worker_class.
    num_workers : int
        Number of remote workers in the worker set.
    """
    def __init__(self,
                 initial_weights,
                 create_algo_instance,
                 create_storage_instance,
                 create_test_envs_instance,
                 create_train_envs_instance,
                 create_actor_critic_instance,
                 worker_remote_config=default_remote_config,
                 num_workers=1):

        self.worker_class = CWorker
        default_remote_config.update(worker_remote_config)
        self.remote_config = default_remote_config
        self.worker_params = {
            "initial_weights": initial_weights,
            "create_algo_instance": create_algo_instance,
            "create_storage_instance": create_storage_instance,
            "create_test_envs_instance": create_test_envs_instance,
            "create_train_envs_instance": create_train_envs_instance,
            "create_actor_critic_instance": create_actor_critic_instance,
        }

        super(CWorkerSet, self).__init__(
            worker=self.worker_class,
            worker_params=self.worker_params,
            worker_remote_config=self.remote_config,
            num_workers=num_workers,
            add_local_worker=False)

    @classmethod
    def worker_set_factory(cls,
                           create_test_envs_instance,
                           create_train_envs_instance,
                           create_actor_critic_instance,
                           worker_remote_config=default_remote_config,
                           num_workers=1):
        """
        Returns a function to create new CWorkerSet instances.

        Parameters
        ----------
        create_test_envs_instance : func
            A function to create test environments.
        create_train_envs_instance : func
            A function to create train environments.
        create_actor_critic_instance : func
            A function that creates a policy.
        worker_remote_config : dict
            Ray resource specs for the remote workers.
        num_workers : int
            Number of remote workers in the worker set.

        Returns
        -------
        create_algo_instance : func
            creates a new CWorkerSet class instance.
        """
        def create_worker_set_instance(
                initial_weights,
                create_algo_instance,
                create_storage_instance):
            return cls(num_workers=num_workers,
                       initial_weights=initial_weights,
                       worker_remote_config=worker_remote_config,
                       create_algo_instance=create_algo_instance,
                       create_storage_instance=create_storage_instance,
                       create_test_envs_instance=create_test_envs_instance,
                       create_train_envs_instance=create_train_envs_instance,
                       create_actor_critic_instance=create_actor_critic_instance)
        return create_worker_set_instance

