import ray

def broadcast_message(key, message):
    ray.worker.global_worker.redis_client.set(key, message)

def check_message(key):
    return ray.worker.global_worker.redis_client.get(key)
