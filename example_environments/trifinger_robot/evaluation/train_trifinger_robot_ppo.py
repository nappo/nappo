"""
Training script for PPO CENTRAL BASIC:

    L   D   S   A
   --- --- --- ---
R | X |   | X |   |
   --- --- --- ---
G | X |   | X |   |
   --- --- --- ---

(L) Local (S) Synchronous (G) Gradients, (L) Local (S) Synchronous (R) Rollouts.

from https://arxiv.org/pdf/1707.06347.pdf

"""

import os
import ray
import sys
import time
import json
import argparse
import numpy as np

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../..')

from nappo import utils
from nappo.core import Algo, Policy, RolloutStorage
from envs import make_vec_envs_func, make_trifinger_robot_env
from nappo.distributed import ppo


def main():

    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    utils.cleanup_log_dir(args.log_dir)
    args_dict = vars(args)
    json.dump(args_dict, open(os.path.join(args.log_dir, "training_arguments.json"), "w"), indent=4)

    # 1. Define Single Env
    env_make = make_trifinger_robot_env(log_dir=args.log_dir, frame_skip=args.frame_skip)

    # 2. Define Vector of Envs
    envs, action_space, obs_space = make_vec_envs_func(
        env_make, args.num_env_processes, args.frame_stack)

    # 3. Define RL Policy
    policy = Policy.create_policy_func(
        args.cnn, obs_space, action_space,
        base_kwargs={'recurrent': args.recurrent_policy},
        restart_model=args.restart_model)

    # 4. Define RL training algorithm
    agent = Algo.create_algo_func(
        lr=args.lr, eps=args.eps, num_epochs=args.ppo_epoch, clip_param=args.clip_param,
        entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm, num_mini_batch=args.num_mini_batch,
        use_clipped_value_loss=args.use_clipped_value_loss)

    # 5. Define rollouts storage
    rollouts = RolloutStorage.create_rollout_func(
        num_steps=args.num_steps, num_processes=args.num_env_processes,
        gamma=args.gamma, use_gae=args.use_gae, gae_lambda=args.gae_lambda)

    # 6. Define worker
    worker = ppo.Worker(
        index_worker=0, create_envs_func=envs, create_policy_func=policy,
        create_rollout_func=rollouts, create_remote_agent_func=agent,
        device=args.device)

    # 7. Define learner
    learner = ppo.Learner(worker, target_steps=args.num_env_steps, tb_log_dir=args.log_dir)

    # 8. Define train loop
    start_time = time.time()
    iteration_time = time.time()

    while not learner.done():

        learner.step()
        if learner.iterations % args.log_interval == 0 and learner.iterations != 0:
            learner.print_info(iteration_time)
            iteration_time = time.time()

        if learner.iterations % args.save_interval == 0 and learner.iterations != 0:
            save_name = worker.save_model(
                os.path.join(args.log_dir, "trifinger_robot.state_dict"),
                learner.iterations)
            args_dict.update({"latest_model": save_name})
            json.dump(args_dict, open(os.path.join(args.log_dir, "training_arguments.json"), "w"), indent=4)
            # json.dump(learner.metric_arrays, open( os.path.join(args.log_dir, "metrics_arrays.json"), "w"), indent=4)

        if args.max_time != -1 and (time.time() - start_time) > args.max_time:
            break

    print("Finished!")
    sys.exit()

def get_args():
    parser = argparse.ArgumentParser(description='RL', conflict_handler='resolve')
    parser.add_argument(
        '--env-id', type=str, default=None, help='Gym environment id (default None)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps', type=float, default=1e-5, help='Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--num-env-processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--max-time', type=int, default=-1, help='stop script after this amount of time in seconds (default: no limit)')
    parser.add_argument(
        '--num-steps', type=int, default=20000, help='number of forward steps in PPO (default: 20000)')
    parser.add_argument(
        '--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval', type=int, default=1, help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval', type=int, default=100, help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--num-env-steps', type=int, default=10e7, help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--max-time', type=int, default=-1, help='stop script after this amount of time in seconds (default: no limit)')
    parser.add_argument(
        '--log-dir', default='/tmp/ppo/', help='directory to save agent logs (default: /tmp/ppo)')
    parser.add_argument(
        '--use-proper-time-limits', action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument(
        '--use_clipped_value_loss', action='store_true', default=False, help='clip value loss update')
    parser.add_argument(
        '--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument(
        '--frame-skip', type=int, default=0, help='Number of frame to skip for each action')
    parser.add_argument(
        '--frame-stack', type=int, default=4, help='Number of frame to stack in observation')
    parser.add_argument(
        '--restart-model', default='', help='Restart training using the model given')
    parser.add_argument(
        '--cnn', default='Fixup', help='Type of cnn. Options are CNN, Fixup')
    parser.add_argument(
        '--device', default='cuda:0', help='Device to run the learner on')
    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
