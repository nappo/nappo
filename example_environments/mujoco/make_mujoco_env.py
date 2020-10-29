import gym

def make_mujoco_train_env(env_id, seed=0):
    env = gym.make(env_id)
    env.seed(seed)
    return env

def make_mujoco_test_env(env_id, seed=0):
    env = gym.make(env_id)
    env.seed(seed)
    return env