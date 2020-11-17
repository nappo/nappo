import os
import ray
import sys
import time
import json
import argparse

from nappo import utils
from nappo import Learner
from nappo.core.algos import PPO
from nappo.core.envs import vec_envs_factory
from nappo.core.storage import OnPolicyGAEBuffer
from nappo.distributed_schemes.scheme_dadaca import Workers
from nappo.core.models import OnPolicyActorCritic, get_model
from nappo.envs import make_pybullet_train_env, make_pybullet_test_env


def main():

    args = get_args()
    args_dict = vars(args)
    utils.cleanup_log_dir(args.log_dir)
    json.dump(args_dict, open(os.path.join(args.log_dir, "training_arguments.json"), "w"), indent=4)

    if args.cluster:
        ray.init(address="auto")
    else:
        ray.init()

    resources = ""
    for k, v in ray.cluster_resources().items():
        resources += "{} {}, ".format(k, v)
    print(resources[:-2], flush=True)

    # 1. Define Train Vector of Envs
    create_train_envs, action_space, obs_space = vec_envs_factory(
        num_processes=args.num_env_processes, log_dir=args.log_dir,
        env_fn=make_pybullet_train_env, env_kwargs={
            "env_id": args.env_id,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack})

    # 2. Define Test Vector of Envs (Optional)
    create_test_envs, _, _ = vec_envs_factory(
        num_processes=args.num_env_processes, log_dir=args.log_dir,
        env_fn=make_pybullet_test_env, env_kwargs={
            "env_id": args.env_id,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack})

    # 3. Define RL Policy
    create_actor_critic = OnPolicyActorCritic.actor_critic_factory(
        obs_space, action_space,
        feature_extractor_network=get_model(args.nn),
        recurrent_policy=args.recurrent_policy,
        restart_model=args.restart_model)

    # 4. Define RL training algorithm
    create_algo = PPO.algo_factory(
        lr=args.lr, eps=args.eps, num_epochs=args.ppo_epoch, clip_param=args.clip_param,
        entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm, num_mini_batch=args.num_mini_batch,
        use_clipped_value_loss=args.use_clipped_value_loss, gamma=args.gamma)

    # 5. Define rollouts storage
    create_storage = OnPolicyGAEBuffer.storage_factory(
        size=args.num_steps, gae_lambda=args.gae_lambda)

    # 6. Define workers
    workers = Workers(
        create_algo_instance=create_algo,
        create_storage_instance=create_storage,
        create_train_envs_instance=create_train_envs,
        create_test_envs_instance=create_test_envs,
        create_actor_critic_instance=create_actor_critic,
        num_col_workers=args.num_col_workers, col_worker_remote_config={"num_gpus": 0.25},
        num_grad_workers=args.num_grad_workers, grad_worker_remote_config={"num_gpus": 0.25})

    # 7. Define learner
    learner = Learner(workers, target_steps=args.num_env_steps, log_dir=args.log_dir)

    # 8. Define train loop
    iterations = 0
    start_time = time.time()
    while not learner.done():

        learner.step()
        if iterations % args.log_interval == 0:
            learner.print_info()

        if iterations % args.save_interval == 0:
            save_name = learner.save_model()
            args_dict.update({"latest_model": save_name})
            args_path = os.path.join(args.log_dir, "training_arguments.json")
            json.dump(args_dict, open(args_path, "w"), indent=4)

        if args.max_time != -1 and (time.time() - start_time) > args.max_time:
            break

        iterations += 1

    print("Finished!")
    sys.exit()

def get_args():
    parser = argparse.ArgumentParser(description='RL', conflict_handler='resolve')

    # Environment specs
    parser.add_argument(
        '--env-id', type=str, default=None,
        help='Gym environment id (default None)')
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action')
    parser.add_argument(
        '--frame-stack', type=int, default=4,
        help='Number of frame to stack in observation')

    # PPO specs
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps', type=float, default=1e-5,
        help='Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae', action='store_true', default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda', type=float, default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef', type=float, default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef', type=float, default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--use_clipped_value_loss', action='store_true', default=False,
        help='clip value loss update')
    parser.add_argument(
        '--num-steps', type=int, default=20000,
        help='number of forward steps in PPO (default: 20000)')
    parser.add_argument(
        '--ppo-epoch', type=int, default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch', type=int, default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param', type=float, default=0.2,
        help='ppo clip parameter (default: 0.2)')

    # Feature extractor model specs
    parser.add_argument(
        '--nn', default='MLP',
        help='Type of nn. Options are MLP, CNN, Fixup')
    parser.add_argument(
        '--restart-model', default='',
        help='Restart training using the model given')
    parser.add_argument(
        '--device', default='cuda:0', help='Device to run the learner on')
    parser.add_argument(
        '--recurrent-policy', action='store_true', default=False,
        help='use a recurrent policy')

    # Scheme specs
    parser.add_argument(
        '--num-env-processes', type=int, default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-col-workers', type=int, default=1,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--num-grad-workers', type=int, default=1,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--cluster', action='store_true',
        default=False, help='script is running in a cluster')

    # General training specs
    parser.add_argument(
        '--num-env-steps', type=int, default=10e7,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--max-time', type=int, default=-1,
        help='stop script after this amount of time in seconds (default: no limit)')
    parser.add_argument(
        '--log-interval', type=int, default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval', type=int, default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--log-dir', default='/tmp/ppo/',
        help='directory to save agent logs (default: /tmp/ppo)')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
