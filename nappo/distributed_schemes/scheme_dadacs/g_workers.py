import ray
import time
import torch
from ..base.worker import Worker as W
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config
from ..utils import TaskPool, ray_get_and_free


class GWorker(W):
    """
    Worker class handling gradient computation.

    This class wraps an actor_critic instance, a storage class instance and a
    worker set of remote data collection workers. It receives data from the
    collection workers and computes gradients following a logic defined in
    function self.step(), which will be called from the Learner class.

    Parameters
    ----------
    index_worker : int
        Worker index.
    create_algo_instance : func
        A function that creates an algorithm class.
    create_storage_instance : func
        A function that create a rollouts storage.
    create_actor_critic_instance : func
        A function that creates a policy.
    initial_weights : ray object ID
        Initial model weights.
    max_collect_requests_pending : int
        maximum number of collection tasks simultaneously scheduled to each
        collection worker.

    Attributes
    ----------
    index_worker : int
        Index assigned to this worker.
    actor_critic : nn.Module
        An actor_critic class instance.
    algo : an algorithm class
        An algorithm class instance.
    storage : a rollout storage class
        A Storage class instance.
    iter : int
        Times gradients have been computed and sent.
    ac_version : int
        Number of times the current actor critic version been has been updated.
    latest_weights : ray object ID
        Last received model weights.
    collector_tasks : TaskPool
        Tracks the status of in-flight actor collection tasks.
    """

    def __init__(self,
                 index_worker,
                 create_algo_instance,
                 create_storage_instance,
                 create_actor_critic_instance,
                 create_collection_worker_set_instance,
                 max_collect_requests_pending=2,
                 initial_weights=None):

        super(GWorker, self).__init__(index_worker)

        # worker should only see one GPU or None
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Create Actor Critic instance
        self.actor_critic = create_actor_critic_instance(device)
        self.actor_critic.to(device)

        # Create Algorithm instance
        self.algo = create_algo_instance(self.actor_critic, device)

        # Define counters and other attributes
        self.iter = 0
        self.ac_version = 0

        if initial_weights:  # if remote worker

            # Create CWorkerSet instance
            self.c_workers = create_collection_worker_set_instance(
                initial_weights, create_algo_instance, create_storage_instance,
                ).remote_workers()

            # Create Storage instance and set world initial state
            self.storage = create_storage_instance(device)

            # Print worker information
            self.print_worker_info()

            # Set initial weights to collector workers and start data collection
            self.latest_weights = initial_weights
            self.collector_tasks = TaskPool()
            for collector_worker in self.c_workers:
                for _ in range(max_collect_requests_pending):
                    collector_worker.set_weights.remote(self.latest_weights)
                    self.collector_tasks.add(
                        collector_worker, collector_worker.collect_data.remote())

    def compute_gradients(self, batch):
        """
        Calculate actor critic gradients and update networks.

        Parameters
        ----------
        batch : dict
            data batch containing all required tensors to compute algo loss.

        Returns
        -------
        grads: list of tensors
            List of actor_critic gradients.
        info : dict
            Summary dict with relevant gradient-related information.
        """

        t = time.time()
        grads, info = self.algo.compute_gradients(batch)
        info.update({"scheme/seconds_to/compute_grads": time.time() - t})

        return grads, info

    def update_networks(self, gradients):
        """Update Actor Critic model"""
        self.algo.apply_gradients(gradients)

    def step(self):
        """
        Perform logical learning step. Training proceeds receiving data samples
        from collection workers and computations policy gradients.

        Returns
        -------
        grads: list of tensors
            List of actor_critic gradients.
        info : dict
            Summary dict of relevant step information.
        """

        # Collect data and prepare data batches
        if self.iter % (self.algo.num_epochs * self.algo.num_mini_batch) == 0:

            # Wait to remote workers to complete data collection tasks
            for collector_worker, rollouts in self.collector_tasks.completed(blocking_wait=True, max_yield=1):
                new_rollouts = ray_get_and_free(rollouts)
                collector_worker.set_weights.remote(self.latest_weights)
                self.collector_tasks.add(collector_worker, collector_worker.collect_data.remote())

            self.collect_info = new_rollouts["info"]
            self.storage.add_data(new_rollouts["data"])
            self.storage.before_update(self.actor_critic, self.algo)
            self.batches = self.storage.generate_batches(
                self.algo.num_mini_batch, self.algo.mini_batch_size,
                self.algo.num_epochs, self.actor_critic.is_recurrent)

        else:
            collect_info = {}
            collect_info.update({"collected_samples": 0})

        # Compute gradients, get algo info
        grads, info = self.compute_gradients(self.batches.__next__())

        # Add extra information to info dict
        info.update(self.collect_info)
        self.collect_info.update({"collected_samples": 0})
        info.update({"ac_update_num": self.iter})
        info.update({"scheme/metrics/collection_gradient_delay": self.iter - self.collect_info["ac_version"]})

        # Update counter
        self.iter += 1

        return grads, info

    def set_weights(self, weights):
        """
        Update the worker actor_critic version with provided weights.

        weights: dict of tensors
            Dict containing actor_critic weights to be set.
        """
        self.latest_weights = weights
        self.ac_version = weights["update"]
        self.algo.set_weights(weights["weights"])

    def terminate_worker(self):
        """Terminate data collection remote workers"""
        for e in self.c_workers:
            e.terminate_worker.remote()
        ray.actor.exit_actor()

    def update_algo_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of Worker.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        """
        self.algo.update_algo_parameter(parameter_name, new_parameter_value)
        for e in self.c_workers.remote_workers():
            e.update_algo_parameter.remote(parameter_name, new_parameter_value)


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