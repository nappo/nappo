import ray
from .worker import default_remote_config


class WorkerSet:
    """
    Class to better handle the operations of ensembles of Workers.
    Contains common functionality across all worker sets.

    Parameters
    ----------
    worker : func
        A function that creates a worker class.
    worker_params : dict
        Worker class kwargs.
    worker_remote_config : dict
        Ray resource specs for the remote workers.
    num_workers : int
        Num workers replicas in the worker_set.
    add_local_worker : bool
        Whether or not to include have a non-remote worker in the worker set.

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
                 worker,
                 worker_params,
                 worker_remote_config=default_remote_config,
                 num_workers=1,
                 add_local_worker=True):

        self.worker_class = worker
        self.worker_params = worker_params
        self.remote_config = worker_remote_config

        if add_local_worker:
            self._local_worker = self._make_worker(
                self.worker_class, index_worker=0,
                worker_params=self.worker_params)
            self.worker_params.update({
                "initial_weights": ray.put({
                    "update": 0,
                    "weights": self._local_worker.get_weights()})})

        else:
            self._local_worker = None

        self._remote_workers = []
        self.add_workers(num_workers)
        self.num_workers = num_workers

    @staticmethod
    def _make_worker(cls, index_worker, worker_params):
        """
        Create a single worker.

        Parameters
        ----------
        index_worker : int
            Index assigned to remote worker.
        worker_params : dict
            Keyword parameters of the worker_class.

        Returns
        -------
        w : python class
            An instance of worker class cls
        """
        w = cls(index_worker=index_worker, **worker_params)
        return w

    def add_workers(self, num_workers):
        """
        Create and add a number of remote workers to this worker set.

        Parameters
        ----------
        num_workers : int
            Number of remote workers to create.
        """
        cls = self.worker_class.as_remote(**self.remote_config).remote
        self._remote_workers.extend([
            self._make_worker(cls, index_worker=i + 1, worker_params=self.worker_params)
            for i in range(num_workers)])

    def local_worker(self):
        """Return local worker"""
        return self._local_worker

    def remote_workers(self):
        """Returns list of remote workers"""
        return self._remote_workers

    def stop(self):
        """Stop all remote workers"""
        for w in self.remote_workers():
            w.__ray_terminate__.remote()
