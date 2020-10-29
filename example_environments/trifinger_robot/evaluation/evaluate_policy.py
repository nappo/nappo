#!/usr/bin/env python3
"""Example evaluation script to evaluate a policy.

This is an example evaluation script for evaluating a "RandomPolicy".  Use this
as a base for your own script to evaluate your policy.  All you need to do is
to replace the `RandomPolicy` and potentially the Gym environment with your own
ones (see the TODOs in the code below).

This script will be executed in an automated procedure.  For this to work, make
sure you do not change the overall structure of the script!

This script expects the following arguments in the given order:
 - Difficulty level (needed for reward computation)
 - initial pose of the cube (as JSON string)
 - goal pose of the cube (as JSON string)
 - file to which the action log is written

It is then expected to initialize the environment with the given initial pose
and execute exactly one episode with the policy that is to be evaluated.

When finished, the action log, which is created by the TriFingerPlatform class,
is written to the specified file.  This log file is crucial as it is used to
evaluate the actual performance of the policy.
"""


import sys
import gym
import numpy as np

from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.tasks import move_cube




class RandomPolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()


def main():
    try:
        difficulty = int(sys.argv[1])
        initial_pose_json = sys.argv[2]
        goal_pose_json = sys.argv[3]
        output_file = sys.argv[4]
    except IndexError:
        print("Incorrect number of arguments.")
        print(
            "Usage:\n"
            "\tevaluate_policy.py <difficulty_level> <initial_pose>"
            " <goal_pose> <output_file>"
        )
        sys.exit(1)

    # the poses are passes as JSON strings, so they need to be converted first
    initial_pose = move_cube.Pose.from_json(initial_pose_json)
    goal_pose = move_cube.Pose.from_json(goal_pose_json)

    # create a FixedInitializer with the given values
    initializer = cube_env.FixedInitializer(
        difficulty, initial_pose, goal_pose
    )

    # TODO: Replace with your environment if you used a custom one.
    import os
    import json
    import torch
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
    from envs import make_real_robot_test_env
    trained_dir = "/tmp/trifinger_robot_single_threaded_sac_4/"
    log_args = json.load(open(os.path.join(trained_dir, "training_arguments.json")))

    # 1. Define Single Env
    env = make_real_robot_test_env(
        initializer=initializer, frame_skip=log_args["frame_skip"],
        visualization=False, frame_stack=log_args["frame_stack"])

    # TODO: Replace this with your model
    # Note: You may also use a different policy for each difficulty level (difficulty)
    from nappo.core.models import OffPolicyActorCritic
    actor_critic = OffPolicyActorCritic.actor_critic_factory(
        env.observation_space, env.action_space,
        feature_extractor_kwargs={"hidden_sizes":[256, 512, 512, 512]},
        create_double_q_critic=True)(torch.device("cpu"))
    if "restart_model" in log_args.keys():
        actor_critic.load_state_dict(torch.load(
            os.path.join(trained_dir, log_args["restart_model"].split("/")[-1]),
            map_location=torch.device("cpu")))

    rnn_hxs = torch.zeros(actor_critic.recurrent_hidden_state_size)
    done = np.zeros((1))

    # Execute one episode.  Make sure that the number of simulation steps
    # matches with the episode length of the task.  When using the default Gym
    # environment, this is the case when looping until is_done == True.  Make
    # sure to adjust this in case your custom environment behaves differently!
    is_done = False
    observation = env.reset()
    accumulated_reward = 0
    num_steps = 0

    while not is_done:

        observation = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        done = torch.as_tensor(done, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, action, _, rnn_hxs, _ = actor_critic.get_action(
                observation, rnn_hxs, done, deterministic=True)

        observation, reward, is_done, info = env.step(action.detach().squeeze(0).numpy())
        accumulated_reward += reward
        num_steps += 1

    print("Accumulated reward: {}".format(accumulated_reward))

    # store the log for evaluation
    env.platform.store_action_log(output_file)


if __name__ == "__main__":
    main()
