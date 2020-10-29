import gym

def make_pybullet_train_env(env_id, seed=0):
    env = gym.make(env_id)
    env.seed(seed)
    return env

def make_pybullet_test_env(env_id, seed=0):
    env = gym.make(env_id)
    env.seed(seed)
    return env
