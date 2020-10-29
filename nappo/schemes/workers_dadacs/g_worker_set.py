import ray
from .g_worker import GWorker
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config

class GWorkerSet(WS):
    """
    Class to better handle the operations of ensembles of workers.

    Parameters
    ----------
    create_algo_instance : func
        A function that creates an algorithm class.
    create_storage_instance : func
        A function that create a rollouts storage.
    create_actor_critic_instance : func
        A function that creates a policy.
    create_collection_worker_set_instance : func
        A function that creates worker sets of data collection workers.
    worker_remote_config : dict
        Ray resource specs for the remote workers.
    max_collector_workers_requests_pending : int
        maximum number of collection tasks to simultaneously scheduled to
        collection workers.
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
                 create_algo_instance,
                 create_storage_instance,
                 create_actor_critic_instance,
                 create_collection_worker_set_instance,
                 worker_remote_config=default_remote_config,
                 max_collector_workers_requests_pending=2,
                 num_workers=1):
        """
        Initialize a WorkerSet.

        Arguments:
            create_algo_instance (func): a function that creates an algorithm class.
            create_storage_instance (func): a function that create a rollouts storage.
            create_actor_critic_instance (func): a function that creates a policy.
            max_collector_workers_requests_pending (int): maximum number of collect() tasks to schedule for each collection worker.
            worker_remote_config (dict): ray resources specs for the gradient workers.
            num_workers (int): number of gradient computing workers.
        """

        self.worker_class = GWorker
        default_remote_config.update(worker_remote_config)
        self.remote_config = default_remote_config
        self.worker_params = {
            "create_algo_instance": create_algo_instance,
            "create_storage_instance": create_storage_instance,
            "create_actor_critic_instance": create_actor_critic_instance,
            "max_collect_requests_pending": max_collector_workers_requests_pending,
            "create_collection_worker_set_instance": create_collection_worker_set_instance,
        }

        super(GWorkerSet, self).__init__(
            worker=self.worker_class,
            worker_params=self.worker_params,
            worker_remote_config=self.remote_config,
            num_workers=num_workers)