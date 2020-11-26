from .c_worker import CWorker
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config


class CWorkerSet(WS):
    """
    Class to better handle the operations of ensembles of CWorkers.

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
                 num_workers,
                 algo_factory,
                 actor_factory,
                 storage_factory,
                 train_envs_factory,
                 local_device=None,
                 initial_weights=None,
                 test_envs_factory=lambda x, y, c: None,
                 worker_remote_config=default_remote_config):

        self.worker_class = CWorker
        default_remote_config.update(worker_remote_config)
        self.remote_config = default_remote_config
        self.worker_params = {
            "algo_factory": algo_factory,
            "storage_factory": storage_factory,
            "test_envs_factory": test_envs_factory,
            "train_envs_factory": train_envs_factory,
            "actor_factory": actor_factory,
        }

        self.num_workers = num_workers
        super(CWorkerSet, self).__init__(
            worker=self.worker_class,
            local_device=local_device,
            num_workers=self.num_workers,
            initial_weights=initial_weights,
            worker_params=self.worker_params,
            worker_remote_config=self.remote_config)

    @classmethod
    def create_factory(cls,
                       num_workers,
                       algo_factory,
                       actor_factory,
                       storage_factory,
                       test_envs_factory,
                       train_envs_factory,
                       col_worker_resources=default_remote_config):
        """
        Returns a function to create new CWorkerSet instances.

        Parameters
        ----------
        algo_factory : func
            A function that creates an algorithm class.
        actor_factory : func
            A function that creates a policy.
        storage_factory : func
            A function that create a rollouts storage.
        test_envs_factory : func
            A function to create test environments.
        train_envs_factory : func
            A function to create train environments.
        worker_remote_config : dict
            Ray resource specs for the remote workers.
        num_workers : int
            Number of remote workers in the worker set.

        Returns
        -------
        create_algo_instance : func
            creates a new CWorkerSet class instance.
        """

        def collection_worker_set_factory(device, initial_weights):
            return cls(
                local_device=device,
                num_workers=num_workers,
                algo_factory=algo_factory,
                actor_factory=actor_factory,
                storage_factory=storage_factory,
                initial_weights=initial_weights,
                test_envs_factory=test_envs_factory,
                train_envs_factory=train_envs_factory,
                worker_remote_config=col_worker_resources)

        return collection_worker_set_factory