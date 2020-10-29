import os
import sys
import json
import torch
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../..')

from nappo.core.models import OffPolicyActorCritic
from envs import make_real_robot_test_env

def main():

    args = get_args()
    args.det = not args.non_det
    device = torch.device(args.device)
    torch.set_num_threads(1)

    log_args = json.load(open(os.path.join(args.trained_dir, "training_arguments.json")))

    # 1. Define Single Env
    env = make_real_robot_test_env(
        frame_skip=log_args["frame_skip"], visualization=True, frame_stack=log_args["frame_stack"])

    actor_critic = OffPolicyActorCritic.actor_critic_factory(
        env.observation_space, env.action_space,
        feature_extractor_kwargs={"hidden_sizes":[256, 512, 512, 512]},
        create_double_q_critic=True)(torch.device("cpu"))

    if "restart_model" in log_args.keys():
        actor_critic.load_state_dict(torch.load(
            os.path.join(args.trained_dir, log_args["restart_model"].split("/")[-1]),
            map_location=device))

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

        observation, reward, is_done, info = env.step(
            action.detach().squeeze(0).numpy())

        print("step: {}".format(num_steps))

        accumulated_reward += reward
        num_steps += 1

    print("Accumulated reward: {}".format(accumulated_reward))


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--trained-dir', default='/tmp/trifinger_robot/', help='directory where train logs were saved (default: /tmp/trifinger_robot)')
    parser.add_argument(
        '--log-dir', default='/tmp/trifinger_robot_ppo/', help='directory where policy logs were saved (default: /tmp/trifinger_robot_ppo)')
    parser.add_argument(
        '--device', default='cpu', help='Device to run on')
    parser.add_argument(
        '--non-det', action='store_true', default=False, help='whether to use a non-deterministic policy')
    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()