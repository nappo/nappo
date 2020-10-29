import sys
from .atari.make_atari_env import make_atari_train_env, make_atari_test_env
from .mujoco.make_mujoco_env import make_mujoco_train_env, make_mujoco_test_env
from .pybullet.make_pybullet_env import make_pybullet_train_env, make_pybullet_test_env
from .trifinger_robot.make_trifinger_robot_env import make_real_robot_train_env, make_real_robot_test_env
from .trifinger_robot.her_function import her_function_robot

# incompatible imports, so need to check which one was imported - check script name
if "animal" in sys.argv[0]:
    from .animal_olympics.make_animal_env import make_animal_env
if "obstacle" in sys.argv[0]:
    from .obstacle_tower.make_obstacle_env import make_obstacle_env
