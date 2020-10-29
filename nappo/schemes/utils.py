import ray
import time
from collections import deque

FREE_DELAY_S = 10.0
MAX_FREE_QUEUE_SIZE = 100
_last_free_time = 0.0
_to_free = []

def ray_get_and_free(object_ids):
    """
    Call ray.get and then queue the object ids for deletion.
    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.

    Adapted from https://github.com/ray-project/ray/blob/master/rllib/utils/memory.py

    Parameters
    ----------
    object_ids : ObjectID|List[ObjectID]
        Object ids to fetch and free.

    Returns
    -------
    result : python objects
        The result of ray.get(object_ids).
    """

    global _last_free_time
    global _to_free

    result = ray.get(object_ids)
    if type(object_ids) is not list:
        object_ids = [object_ids]
    _to_free.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_to_free) > MAX_FREE_QUEUE_SIZE
        or now - _last_free_time > FREE_DELAY_S):
        ray.internal.free(_to_free)
        _to_free = []
        _last_free_time = now

    return result


class TaskPool:
    """
    Class to track the status of many in-flight actor tasks.

    Adapted from https://github.com/ray-project/ray/blob/master/rllib/utils/actors.py
    """

    def __init__(self):
        self._obj_task_to_worker = {}
        self.fetching = deque()

    def add(self, worker, obj_id):
        self._obj_task_to_worker[obj_id] = worker

    def completed(self, blocking_wait=False, max_yield=999):
        """ Yield completed task objects."""
        pending = list(self._obj_task_to_worker)
        if pending:
            ready, _ = ray.wait(pending, num_returns=max_yield if max_yield < len(pending) else len(pending), timeout=10)

            if not ready and blocking_wait:
                while not ready:
                    ready, _ = ray.wait(pending, num_returns=1, timeout=10.0)

            for obj_id in ready:
                yield self._obj_task_to_worker.pop(obj_id), obj_id

def average_gradients(grads_list):
    """
    Averages gradients coming from distributed workers.

    Parameters
    ----------
    grads_list : list of lists of tensors
        List of actor_critic gradients from different workers.
    Returns
    -------
    avg_grads : list of tensors
        Averaged actor_critic gradients.
    """
    if len(grads_list) == 1:
        return grads_list[0]
    avg_grads = [
        sum(d[grad] for d in grads_list) / len(grads_list) if
        grads_list[0][grad] is not None else 0.0
        for grad in range(len(grads_list[0]))]
    return avg_grads


