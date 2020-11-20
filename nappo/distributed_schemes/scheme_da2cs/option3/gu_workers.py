import os
import ray
import time
import torch
from shutil import copy2
from functools import partial
from collections import defaultdict, deque
from ray.services import get_node_ip_address
from nappo.distributed_schemes.base.worker import Worker as W
from nappo.distributed_schemes.base.worker_set import WorkerSet as WS
from nappo.distributed_schemes.utils import TaskPool, ray_get_and_free
from nappo.distributed_schemes.base.worker import default_remote_config


class GUWorker(W):
    """
    Update worker. Handles actor updates.

    This class coordinates sequential central actor optimization, using
    rollouts collected by distributed workers to compute gradients and update
    the models.

    Parameters
    ----------
    col_workers : WorkerSet
        Set of workers collecting and sending rollouts to the UWorker.
    device : torch.device
        CPU or specific GPU to use for computation.
    broadcast_interval : int
        After how many central updates, model weights should be broadcasted to
        remote collection workers.
    max_collect_requests_pending : int
        maximum number of collection tasks simultaneously scheduled to each
        collection worker.

    Attributes
    ----------
    ps : Worker
        Local worker that acts as a parameter server.
    latest_weights : ray object ID
    col_workers : list of Workers
        Set of workers collecting and sending rollouts to the UWorker.
    num_updates : int
        Number of times the actor model has been updated.
    num_workers : int
        number of remote workers computing gradients.
    broadcast_interval : int
        After how many central updates, model weights should be broadcasted to
        remote collection workers.
    """
    def __init__(self,
                 index_worker,
                 collection_workers_factory,
                 broadcast_interval=1,
                 max_collect_requests_pending=2,
                 updater_queue_size = None,
                 initial_weights=None):

        super(GUWorker, self).__init__(index_worker)

        # worker should only see one GPU or None
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        workers = collection_workers_factory(initial_weights)

        self.ps = workers.local_worker()
        self.col_workers = workers.remote_workers()

        self.ps.actor.to(device) # necessary?
        self.latest_weights = ray.put({"update": 0, "weights": self.ps.get_weights()})
        self.num_workers = len(self.col_workers)
        self.broadcast_interval = broadcast_interval

        # Check remote workers exist
        if len(self.col_workers) == 0:
            raise ValueError("""At least 1 data collection worker required""")

        # Counters and metrics
        self.num_updates = 0
        self.num_sent_since_broadcast = 0
        self.metrics = defaultdict(partial(deque, maxlen=100))

        # Start collecting data
        self.collector_tasks = TaskPool()
        for ev in self.col_workers:
            for _ in range(max_collect_requests_pending):
                ev.set_weights.remote(self.latest_weights)
                self.collector_tasks.add(ev, ev.collect_data.remote())

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
        grads, info = self.ps.algo.compute_gradients(batch)
        compute_grads_t = time.time() - t

        ###### ALLREDUCE ######################################################

        t = time.time()
        if torch.cuda.is_available():
            for g in grads:
                torch.distributed.all_reduce(g, op=torch.distributed.ReduceOp.SUM)
        else:
            torch.distributed.all_reduce_coalesced(grads, op=torch.distributed.ReduceOp.SUM)

        for p in self.actor.parameters():
            if p.grad is not None:
                p.grad /= self.distributed_world_size
        avg_grads_t = time.time() - t

        #######################################################################

        info.update({"scheme/seconds_to/compute_grads_t": compute_grads_t})
        info.update({"scheme/seconds_to/avg_grads_t": avg_grads_t})

        return info

    def update_networks(self):
        """Update Actor Critic model"""
        self.ps.algo.apply_gradients()

    def should_broadcast(self):
        """Returns whether broadcast() should be called to update weights"""
        return self.num_sent_since_broadcast >= self.broadcast_interval

    def broadcast_new_weights(self):
        """Broadcast a new set of weights from the local worker"""
        self.latest_weights = ray.put({
            "update": self.num_updates,
            "weights": self.ps.get_weights()})
        self.num_sent_since_broadcast = 0

    def step(self):
        """
        Takes a logical optimization step.

        Returns
        -------
        info : dict
            Summary dict of relevant step information.
        """

        n = self.ps.algo.num_epochs * self.ps.algo.num_mini_batch
        if self.num_updates % n == 0:

            # Wait to remote workers to complete data collection tasks
            for e, rollouts in self.collector_tasks.completed(blocking_wait=True,  max_yield=1):

                # Move new collected rollouts to parameter server input queue
                new_rollouts = ray_get_and_free(rollouts)

                # Update counter and broadcast weights to worker if necessary
                self.num_sent_since_broadcast += 1
                if self.should_broadcast():
                    self.broadcast_new_weights()

                # Request more data from worker
                e.set_weights.remote(self.latest_weights)
                self.collector_tasks.add(e, e.collect_data.remote())

            self.ps.storage.add_data(new_rollouts["data"])
            self.rollouts_info = new_rollouts["info"]
            self.ps.storage.before_update(self.ps.actor, self.ps.algo)

            # Prepare data batches
            self.batches = self.ps.storage.generate_batches(
                self.ps.algo.num_mini_batch, self.ps.algo.mini_batch_size,
                self.ps.algo.num_epochs, self.ps.actor.is_recurrent)

        # Compute grads
        info = self.compute_gradients(self.batches.__next__())

        # Apply grads
        self.update_networks()

        # Add extra information to info dict
        info.update(self.rollouts_info)
        info.update({"scheme/metrics/gradient_update_delay": 0})
        info.update({"scheme/metrics/collection_gradient_delay": self.num_updates - self.rollouts_info["ac_version"]})

        # Update counter
        self.num_updates += 1
        self.rollouts_info["collected_samples"] = 0 # count only once

        return info

    def stop(self):
        """Stop remote workers"""
        for e in self.col_workers:
            e.terminate_worker.remote()

    def get_weights(self):
        """Returns current actor.state_dict() weights"""
        return {k: v.cpu() for k, v in self.ps.actor.state_dict().items()}

    def save_model(self, fname):
        """
        Save current version of actor as a torch loadable checkpoint.

        Parameters
        ----------
        fname : str
            Filename given to the checkpoint.

        Returns
        -------
        save_name : str
            Path to saved file.
        """
        torch.save(self.ps.actor.state_dict(), fname + ".tmp")
        os.rename(fname + '.tmp', fname)
        save_name = fname + ".{}".format(self.num_updates)
        copy2(fname, save_name)
        return save_name

    def update_algo_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of Worker.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        """
        self.ps.algo.update_algo_parameter(parameter_name, new_parameter_value)


class GUWorkerSet(WS):
    """
    Class to better handle the operations of ensembles of workers.

    Parameters
    ----------
    algo_factory : func
        A function that creates an algorithm class.
    storage_factory : func
        A function that create a rollouts storage.
    actor_factory : func
        A function that creates a policy.
    collection_workers_factory : func
        A function that creates a sets of data collection workers.
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
                 collection_workers_factory,
                 worker_remote_config=default_remote_config,
                 max_collector_workers_requests_pending=2,
                 broadcast_interval=1,
                 num_workers=1):

        self.worker_class = GUWorker
        default_remote_config.update(worker_remote_config)
        self.remote_config = default_remote_config
        self.worker_params = {
            "collection_workers_factory": collection_workers_factory,
            "max_collect_requests_pending": max_collector_workers_requests_pending,
        }

        super(GUWorkerSet, self).__init__(
            worker=self.worker_class,
            worker_params=self.worker_params,
            worker_remote_config=self.remote_config,
            num_workers=num_workers)

        ip = ray.get(self.remote_workers()[0].get_node_ip.remote())
        port = ray.get(self.remote_workers()[0].find_free_port.remote())
        address = "tcp://{ip}:{port}".format(ip=ip, port=port)
        ray.get([worker.setup_torch_data_parallel.remote(
            address, i, len(self.remote_workers()), "nccl")
                 for i, worker in enumerate(self.remote_workers())])
