import gym
from ..common import FrameStack, FrameSkip

def make_pybullet_train_env(env_id, seed=0, frame_skip=0, frame_stack=1):
    """_"""

    env = gym.make(env_id)
    env.seed(seed)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env

def make_pybullet_test_env(env_id, seed=0, frame_skip=0, frame_stack=1):
    """_"""

    env = gym.make(env_id)
    env.seed(seed)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env
