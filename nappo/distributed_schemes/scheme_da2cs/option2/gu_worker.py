import ray
import time
import torch
import threading
from six.moves import queue
from functools import partial
from collections import defaultdict, deque
from ..utils import TaskPool, ray_get_and_free


class GUWorker:
    """
    Update worker. Handles actor updates.

    This class coordinates sequential central actor optimization, using
    rollouts collected by distributed workers to compute gradients and update
    the models.

    Parameters
    ----------
    workers : WorkerSet
        Set of workers collecting and sending rollouts to the UWorker.
    device : torch.device
        CPU or specific GPU to use for computation.
    broadcast_interval : int
        After how many central updates, model weights should be broadcasted to
        remote collection workers.
    updater_queue_size : int
        Maximum number of data dicts fitting in the updater queue.
    max_collect_requests_pending : int
        Maximum number of collection tasks simultaneously scheduled to each
        collection worker.

    Attributes
    ----------
    local_worker : Worker
        Local worker that acts as a parameter server.
    remote_workers : list of Workers
        Set of workers collecting and sending rollouts.
    num_updates : int
        Number of times the actor model has been updated.
    num_workers : int
        number of remote workers computing gradients.
    inqueue : queue.Queue
        Queue to store the data dicts received and pending to be processed.
    outqueue : queue.Queue
         Queue to store the info dicts resulting from the model update operation.
    """
    def __init__(self,
                 workers,
                 device="cpu",
                 broadcast_interval=1,
                 updater_queue_size=100,
                 max_collect_requests_pending=2):

        self.local_worker = workers.local_worker()
        self.local_worker.actor.to(device)
        self.remote_workers = workers.remote_workers()
        self.num_workers = len(self.remote_workers)

        # Check remote workers exist
        if self.num_workers == 0:
            raise ValueError("""At least 1 data collection worker required""")

        # Queues
        self.inqueue = queue.Queue(maxsize=updater_queue_size)
        self.outqueue = queue.Queue()

        # Counters and metrics
        self.num_sent_since_broadcast = 0
        self.local_worker.num_updates = 0
        self.metrics = defaultdict(partial(deque, maxlen=100))

        # Create CollectorThread
        self.collector = CollectorThread(
            input_queue=self.inqueue,
            local_worker=self.local_worker,
            remote_workers=self.remote_workers,
            broadcast_interval=broadcast_interval,
            max_collect_requests_pending=max_collect_requests_pending)

        # Start CollectorThread
        self.collector.start()

        # Create UpdaterThread
        self.updater = UpdaterThread(
            input_queue=self.inqueue,
            output_queue=self.outqueue,
            local_worker=self.local_worker)

        # Start UpdaterThread
        self.updater.start()

    @property
    def num_updates(self):
        return self.local_worker.num_updates

    def step(self):
        """Collect and returns information from executed training steps."""

        # Check results in parameter server output queue
        step_metrics = defaultdict(float)
        num_outputs = 0

        while self.outqueue.empty():
            pass

        while not self.outqueue.empty():
            info = self.outqueue.get()
            for k, v in info.items(): step_metrics[k] += v
            num_outputs += 1

        # Update info dict
        info = {k: v / num_outputs if k != "collected_samples"
        else v for k, v in step_metrics.items()}

        return info

    def update_algo_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of Worker.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        """
        self.local_worker.algo.update_algo_parameter(parameter_name, new_parameter_value)

    def stop(self):
        """Stop collecting data and updating the local policy."""
        self.collector.stop()
        self.updater.stop()


class CollectorThread(threading.Thread):
    """
    This class receives data from the workers and queues it to the updater queue.


    Parameters
    ----------
    inqueue : queue.Queue
        Queue to store the data dicts received and pending to be processed.
    local_worker : Worker
        Local worker that acts as a parameter server.
    remote_workers : list of Workers
        Set of workers collecting and sending rollouts.
    broadcast_interval : int
        After how many central updates, model weights should be broadcasted to
        remote collection workers.
    max_collect_requests_pending : int
        Maximum number of collection tasks simultaneously scheduled to each
        collection worker.

    Attributes
    ----------
    input_queue : queue.Queue
        Queue to store the data dicts received and pending to be processed.
    local_worker : Worker
        Local worker that acts as a parameter server.
    remote_workers : list of Workers
        Set of workers collecting and sending rollouts.
    broadcast_interval : int
        After how many central updates, model weights should be broadcasted to
        remote collection workers.
    num_sent_since_broadcast : int
        Number of data dicts received since last model weights were broadcasted.
    num_workers : int
        number of remote workers computing gradients.
    collector_tasks : TaskPool
        Task pool to track remote workers in-flight collection tasks.
    stopped : bool
        Whether or not the thread in running.
    """

    def __init__(self,
                 input_queue,
                 local_worker,
                 remote_workers,
                 broadcast_interval=1,
                 max_collect_requests_pending=2):

        threading.Thread.__init__(self)

        self.inqueue = input_queue
        self.local_worker = local_worker
        self.remote_workers = remote_workers
        self.broadcast_interval = broadcast_interval
        self.num_workers = len(self.remote_workers)

        # Counters and metrics
        self.num_sent_since_broadcast = 0
        self.metrics = defaultdict(partial(deque, maxlen=100))

        # Start collecting data
        self.collector_tasks = TaskPool()
        for ev in self.remote_workers:
            for _ in range(max_collect_requests_pending):
                self.collector_tasks.add(ev, ev.collect_data.remote())

        self.stopped = False

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        """
        Continuously collects data from remote workers and puts it
        in the updater queue.
        """

        # Wait to remote workers to complete data collection tasks
        for e, rollouts in self.collector_tasks.completed(blocking_wait=False, max_yield=1):

            # Move new collected rollouts to parameter server input queue
            self.inqueue.put(ray_get_and_free(rollouts))

            # Update counter and broadcast weights to worker if necessary
            self.num_sent_since_broadcast += 1
            if self.should_broadcast():
                self.broadcast_new_weights()

            # Request more data from worker
            self.collector_tasks.add(e, e.collect_data.remote())

    def should_broadcast(self):
        """Returns whether broadcast() should be called to update weights."""
        return self.num_sent_since_broadcast >= self.broadcast_interval

    def broadcast_new_weights(self):
        """Broadcast a new set of weights from the local worker."""
        latest_weights = ray.put({
            "update": self.local_worker.num_updates,
            "weights": self.local_worker.get_weights()})
        for e in self.remote_workers:
            e.set_weights.remote(latest_weights)
        self.num_sent_since_broadcast = 0

    def stop(self):
        """Stop collecting data."""
        self.stopped = True
        for e in self.remote_workers.remote_workers():
            e.terminate_worker.remote()

class UpdaterThread(threading.Thread):
    """
    This class receives data from the workers and continuously updates central actor.

    Parameters
    ----------
    input_queue : queue.Queue
        Queue to store the data dicts received and pending to be processed.
    output_queue : queue.Queue
        Queue to store the info dicts resulting from the model update operation.
    local_worker : Worker
        Local worker that acts as a parameter server.

    Attributes
    ----------
    local_worker : Worker
        Local worker that acts as a parameter server.
    input_queue : queue.Queue
        Queue to store the data dicts received and pending to be processed.
    output_queue : queue.Queue
        Queue to store the info dicts resulting from the model update operation.
    stopped : bool
        Whether or not the thread in running.
    """

    def __init__(self,
                 input_queue,
                 output_queue,
                 local_worker):

        threading.Thread.__init__(self)

        self.stopped = False
        self.inqueue = input_queue
        self.outqueue = output_queue
        self.local_worker = local_worker

    def run(self):
        while not self.stopped:
            self.step()

    def compute_gradients(self, batch):
        """
        Calculate actor gradients.

        Parameters
        ----------
        batch : dict
            data batch containing all required tensors to compute algo loss.

        Returns
        -------
        info : dict
            Summary dict of relevant gradient-related information.
        """
        t = time.time()
        _, info = self.local_worker.algo.compute_gradients(batch)
        info.update({"scheme/seconds_to/compute_grads_t": time.time() - t})
        return info

    def update_networks(self):
        """Update Actor Critic model"""
        self.local_worker.algo.apply_gradients()

    def step(self):
        """
        Continuously pulls data from the input queue, computes gradients,
        updates the local actor model and places information in the
        output queue.
        """

        n = self.local_worker.algo.num_epochs * self.local_worker.algo.num_mini_batch
        if self.local_worker.num_updates % n == 0:

            new_rollouts = self.inqueue.get(timeout=30)
            self.local_worker.storage.add_data(new_rollouts["data"])
            self.rollouts_info = new_rollouts["info"]
            self.local_worker.storage.before_update(
                self.local_worker.actor, self.local_worker.algo)

            # Prepare data batches
            self.batches = self.local_worker.storage.generate_batches(
                self.local_worker.algo.num_mini_batch, self.local_worker.algo.mini_batch_size,
                self.local_worker.algo.num_epochs, self.local_worker.actor.is_recurrent)

        # Compute grads
        info = self.compute_gradients(self.batches.__next__())

        # Apply grads
        self.update_networks()

        # Add extra information to info dict
        info.update(self.rollouts_info)
        info.update({"scheme/metrics/gradient_update_delay": 0})
        info.update({"scheme/metrics/collection_gradient_delay":
        self.local_worker.num_updates - self.rollouts_info["ac_version"]})

        # Update counter
        self.local_worker.num_updates += 1
        self.rollouts_info["collected_samples"] = 0  # count only once

        self.outqueue.put(info)

    def stop(self):
        """Stop updating the local policy."""
        self.stopped = True