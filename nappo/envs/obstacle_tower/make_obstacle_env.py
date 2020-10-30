import os
import obstacle_tower_env
from obstacle_tower_env import ObstacleTowerEnv
from ..common import FrameStack, FrameSkip
from .wrappers import (ReducedActionEnv, BasicObstacleEnv,
                       RewardShapeObstacleEnv, BasicObstacleEnvTest)

# info_keywords=('floor', 'start', 'seed'),

def make_obstacle_train_env(index_worker=0, index_env=0, realtime=False, seed=0, frame_skip=0, frame_stack=1):
    """
    Create train Obstacle Tower Unity3D challenge environment.

    Parameters
    ----------
    index_worker : int
        Index of the worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    seed : int
        Used as base worker id.
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    realtime : bool
        Visualise environment.

    Returns
    -------
    env : gym.Env
        Train environment.
    """
    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    exe = os.path.join(
        os.path.dirname(obstacle_tower_env.__file__),
        'ObstacleTower/obstacletower')

    id = seed + index_worker + index_env
    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=id, greyscale=False,
        docker_training=False, realtime_mode=realtime)

    env = ReducedActionEnv(env)
    env = BasicObstacleEnv(env, max_floor=50, min_floor=0)
    env = RewardShapeObstacleEnv(env)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env

def make_obstacle_test_env(index_worker=0, index_env=0, realtime=False, seed=0, frame_skip=0, frame_stack=1):
    """
    Create test Obstacle Tower Unity3D challenge environment.

    Parameters
    ----------
    index_worker : int
        Index of the worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    seed : int
        Used as base worker id.
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    realtime : bool
        Visualise environment.

    Returns
    -------
    env : gym.Env
        Test environment.
    """
    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    exe = os.path.join(os.path.dirname(obstacle_tower_env.__file__),
                       'ObstacleTower/obstacletower')
    id = seed + index_worker + index_env
    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=id, greyscale=False,
        docker_training=False, realtime_mode=realtime)

    env = ReducedActionEnv(env)
    env = BasicObstacleEnvTest(env, max_floor=50, min_floor=0)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env
