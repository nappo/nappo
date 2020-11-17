import ray
import time
import torch
import threading
from statistics import mean
from six.moves import queue
from functools import partial
from collections import defaultdict, deque
from ..utils import TaskPool, ray_get_and_free


class GUWorker:
    """
    Class to coordinate sequential central actor_critic learning steps from
    rollouts collected by distributed workers.
    """
    def __init__(self,
                 col_workers,
                 create_algo_instance,
                 create_storage_instance,
                 create_actor_critic_instance,
                 device="cpu",
                 countdown=20,
                 broadcast_interval=1,
                 updater_queue_size=2,
                 updater_queue_timeout=10,
                 max_collect_requests_pending=2):

        """Initialize a GUWorker"""

        self.device = device
        self.workers = col_workers
        self.broadcast_interval = broadcast_interval
        self.local_worker = col_workers.local_worker()
        self.remote_workers = col_workers.remote_workers()
        self.num_workers = len(self.workers.remote_workers())
        self.latest_weights = ray.put({"update": 0, "weights": self.workers.local_worker().get_weights()})

        # Check remote workers exist
        if len(self.remote_workers) == 0:
            raise ValueError("""At least 1 data collection worker required""")

        # Counters and metrics
        self.num_samples_collected = 0
        self.num_sent_since_broadcast = 0
        self.countdown = countdown
        self.countdown_started = False
        self.metrics = defaultdict(partial(deque, maxlen=100))


        # Create CollectorThread
        self.collector = CollectorThread(
            local_worker=self.local_worker,
            remote_workers=self.remote_workers,
            updater_queue_size=updater_queue_size,
            updater_queue_timeout=updater_queue_timeout,
        )

        # Start CollectorThread
        self.collector.start()

        # Create UpdaterThread
        self.updater = UpdaterThread(
            device=torch.device(device),
            actor_critic=self.local_worker.actor_critic,
            updater_queue_size=updater_queue_size,
            updater_queue_timeout=updater_queue_timeout,
            create_algo_instance=create_algo_instance,
            create_storage_instance=create_storage_instance,
        )

        # Start UpdaterThread
        self.updater.start()

    @property
    def num_updates(self):
        return self.updater.num_updates

    def step(self):
        """Takes a logical optimization step."""

        # Check results in parameter server output queue
        step_metrics = defaultdict(float)
        while not self.local_worker.updater.outqueue.empty():
            info = self.local_worker.updater.outqueue.get()
            for k, v in info.items(): step_metrics[k] += v

        # Update info dict
        info = {k: v / self.num_workers for k, v in step_metrics.items()}

        return info

    def stop(self):
        """Stop remote workers"""
        self.local_worker.updater.stopped = True
        for e in self.workers.remote_workers():
            e.terminate_worker.remote()

    def get_weights(self):
        """_"""
        return {k: v.cpu() for k, v in self.updater.actor_critic.state_dict().items()}

    def adjust_algo_parameter(self, parameter_name, new_parameter_value):
        """_"""
        if hasattr(self.updater.algo, parameter_name):
            setattr(self.updater.algo, parameter_name, new_parameter_value)

class CollectorThread(threading.Thread):
    """
    This class receives data from the workers and queues it to the updater queue.
    """

    def __init__(self,
                 local_worker,
                 remote_workers,
                 updater_queue_size,
                 updater_queue_timeout,
                 countdown=20,
                 broadcast_interval=1,
                 max_collect_requests_pending=2):


        """
        Initialize a ParameterServerLearner.
        Arguments:
        """

        threading.Thread.__init__(self)

        self.local_worker = local_worker
        self.remote_workers = remote_workers
        self.broadcast_interval = broadcast_interval
        self.num_workers = len(self.remote_workers)
        self.latest_weights = ray.put({"update": 0, "weights": self.local_worker.get_weights()})

        # Counters and metrics
        self.num_samples_collected = 0
        self.num_sent_since_broadcast = 0
        self.countdown = countdown
        self.countdown_started = False
        self.metrics = defaultdict(partial(deque, maxlen=100))

        self.learner_queue_timeout = updater_queue_timeout
        self.inqueue = queue.Queue(maxsize=updater_queue_size)
        self.outqueue = queue.Queue()

        # Start collecting data
        self.collector_tasks = TaskPool()
        for ev in self.remote_workers:
            for _ in range(max_collect_requests_pending):
                ev.set_weights.remote(self.latest_weights)
                self.collector_tasks.add(ev, ev.collect.remote())

        self.stopped = False

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        """Takes a logical optimization step."""

        # Wait to remote workers to complete data collection tasks
        for e, rollouts in self.collector_tasks.completed(blocking_wait=False, max_yield=1):

            # Move new collected rollouts to parameter server input queue
            self.local_worker.updater.inqueue.put(ray_get_and_free(rollouts))

            # Update counter and broadcast weights to worker if necessary
            self.num_sent_since_broadcast += 1
            if self.should_broadcast():
                self.broadcast_new_weights()

            # Request more data from worker
            if not self.countdown_started:
                e.set_weights.remote(self.latest_weights)
                self.collector_tasks.add(e, e.collect.remote())

    def should_broadcast(self):
        """Returns whether broadcast() should be called to update weights."""
        return self.num_sent_since_broadcast >= self.broadcast_interval

    def broadcast_new_weights(self):
        """Broadcast a new set of weights from the local worker."""
        self.latest_weights = ray.put({
            "update": self.num_updates,
            "weights": self.local_worker.get_weights()})
        self.num_sent_since_broadcast = 0

class UpdaterThread(threading.Thread):
    """
    This class receives data from the workers and continuously updates central actor_critic.
    """

    def __init__(self,
                 device,
                 actor_critic,
                 updater_queue_size,
                 updater_queue_timeout,
                 create_algo_instance,
                 create_storage_instance):
        """
        Initialize a ParameterServerLearner.
        Arguments:
            device (torch.device): device in which the policy will be stored and where the gradients will be computed.
            optimizer (torch.optimizer): optimizer used to compute gradients and update policy weights.
            updater_queue_size (int): max number of rollouts in the input queue.
            updater_queue_timeout (int): max amount of time the updater will wait to get new rollouts.
            create_remote_agent_func (func): function that creates an agent.
            create_actor_critic_instance (func): function that creates a policy.
        """

        threading.Thread.__init__(self)

        self.device = device
        self.num_updates = 0
        self.actor_critic = actor_critic
        self.learner_queue_timeout = updater_queue_timeout
        self.inqueue = queue.Queue(maxsize=updater_queue_size)
        self.outqueue = queue.Queue()

        # Create Algorithm instance
        self.algo = create_algo_instance(self.actor_critic, self.device)

        # Create Storage instance and set world initial state
        self.storage = create_storage_instance(self.device)

        # Retrieve learning information from algo
        self.num_epochs = self.algo.num_epochs
        self.num_mini_batch = self.algo.num_mini_batch
        self.num_updates = self.algo.num_epochs * self.algo.num_mini_batch

        self.stopped = False

    def run(self):
        while not self.stopped:
            self.step()

    def compute_gradients(self, batch):
        """_"""
        t = time.time()
        _, info = self.algo.compute_gradients(batch)
        info.update({"scheme/seconds_to/compute_grads_t": time.time() - t})
        return info

    def update_networks(self):
        """_"""
        self.algo.apply_gradients()

    def step(self):

        metrics = defaultdict(partial(deque, maxlen=self.num_updates))

        try:
            new_rollouts = self.inqueue.get(timeout=self.learner_queue_timeout)
        except queue.Empty:
            return

        self.storage.add_data(new_rollouts["data"])
        self.storage.before_update(self.actor_critic, self.algo)

        # Prepare data batches
        self.batches = self.storage.batch_generator(
            self.algo.num_mini_batch, self.algo.mini_batch_size,
            self.algo.num_epochs, self.actor_critic.is_recurrent)

        for i in range(self.num_updates):

            # Compute grads
            info = self.compute_gradients(self.batches.__next__())

            # Apply grads
            self.update_networks()

            # Add extra information to info dict
            info.update({"scheme/metrics/gradient_update_delay": 0})
            info.update({"scheme/metrics/collection_gradient_delay": self.num_updates - self.storage.ac_version})

            # Update metrics
            for k, v in info.items():
                metrics[k].append(v)

            # Update counter
            self.num_updates += 1

        # Add extra information to info dict
        metrics = {k: mean(v) for k, v in metrics.items()}
        metrics.update(new_rollouts["info"])
        metrics.update({"num_updates": self.num_updates})

        self.outqueue.put(metrics)