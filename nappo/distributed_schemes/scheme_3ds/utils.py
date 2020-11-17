import ray
import socket
from contextlib import closing

def broadcast_message(key, message):
    """ _"""
    ray.worker.global_worker.redis_client.set(key, message)

def check_message(key):
    """ _"""
    return ray.worker.global_worker.redis_client.get(key)

# -----------------------------------------------------------------------------

def find_free_port():
    """
    from https://github.com/ray-project/ray/blob/master/python/ray/util/sgd/utils.py
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]