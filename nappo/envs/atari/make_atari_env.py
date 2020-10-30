from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from .wrappers import wrap_deepmind
from ..common import FrameStack

def make_atari_train_env(env_id, index_worker, index_env, seed=0, frame_stack=1):
    """_"""

    env = make_atari(env_id)
    env.seed(seed + index_worker * index_env)
    env = wrap_deepmind(
        env, episode_life=True,
        clip_rewards=True,
        scale=False, frame_stack=1)

    if frame_stack > 1:
        env = FrameStack(env, frame_stack)

    return env

def make_atari_test_env(env_id, index_worker, index_env, seed=0, frame_stack=1):
    """_"""

    env = make_atari(env_id)
    env.seed(seed + index_worker * index_env)
    env = wrap_deepmind(
        env, episode_life=False,
        clip_rewards=False,
        scale=False, frame_stack=1)

    if frame_stack > 1:
        env = FrameStack(env, frame_stack)

    return env