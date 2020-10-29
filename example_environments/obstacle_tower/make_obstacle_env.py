import os
import obstacle_tower_env
from obstacle_tower_env import ObstacleTowerEnv
from .wrappers import (ReducedActionEnv, BasicObstacleEnv,
                       RewardShapeObstacleEnv, BasicObstacleEnvTest)

# info_keywords=('floor', 'start', 'seed'),

def make_obstacle_train_env(realtime=False, seed=0):

    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    exe = os.path.join(
        os.path.dirname(obstacle_tower_env.__file__),
        'ObstacleTower/obstacletower')

    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=seed, greyscale=False,
        docker_training=False, realtime_mode=realtime)

    env = ReducedActionEnv(env)
    env = BasicObstacleEnv(env, max_floor=50, min_floor=0)
    env = RewardShapeObstacleEnv(env)

    return env

def make_obstacle_test_env(realtime=False, seed=0):

    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    exe = os.path.join(os.path.dirname(obstacle_tower_env.__file__),
                       'ObstacleTower/obstacletower')
    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=seed, greyscale=False,
        docker_training=False, realtime_mode=realtime)

    env = ReducedActionEnv(env)
    env = BasicObstacleEnvTest(env, max_floor=50, min_floor=0)

    return env
