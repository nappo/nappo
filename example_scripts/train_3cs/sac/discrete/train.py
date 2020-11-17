import os
import sys
import time
import json
import argparse

from nappo import utils
from nappo import Learner
from nappo.core.algos import SAC
from nappo.core.storage import ReplayBuffer
from nappo.core.envs import vec_envs_factory
from nappo.distributed_schemes.scheme_3cs import Worker
from nappo.core.models import OffPolicyActorCritic, get_model
from nappo.envs import make_atari_train_env, make_atari_test_env


def main():

    args = get_args()
    args_dict = vars(args)
    utils.cleanup_log_dir(args.log_dir)
    json.dump(args_dict, open(os.path.join(args.log_dir, "training_arguments.json"), "w"), indent=4)

    # 1. Define Train Vector of Envs
    create_train_envs, action_space, obs_space = vec_envs_factory(
        env_fn=make_atari_train_env,
        env_kwargs={"env_id": args.env_id, "frame_stack":args.frame_stack},
        num_processes=args.num_env_processes, log_dir=args.log_dir,
        info_keywords=('rr', 'rrr', 'lives'))

    # 2. Define Test Vector of Envs (Optional)
    create_test_envs, _, _ = vec_envs_factory(
        env_fn=make_atari_test_env,
        env_kwargs={"env_id": args.env_id, "frame_stack":args.frame_stack},
        num_processes=args.num_env_processes, log_dir=args.log_dir)

    # 3. Define RL training algorithm
    create_algo = SAC.algo_factory(
        lr_pi=args.lr, lr_q=args.lr, lr_alpha=args.lr, initial_alpha=args.alpha,
        gamma=args.gamma, polyak=args.polyak, num_updates=args.num_updates,
        update_every=args.update_every, start_steps=args.start_steps,
        mini_batch_size=args.mini_batch_size)

    # 4. Define RL Policy
    create_actor_critic = OffPolicyActorCritic.actor_critic_factory(
        obs_space, action_space,
        recurrent_policy=args.recurrent_policy,
        feature_extractor_network=get_model(args.nn),
        restart_model=args.restart_model)

    # 5. Define rollouts storage
    create_storage = ReplayBuffer.storage_factory(size=args.buffer_size)

    # 6. Define worker
    worker = Worker(
        index_worker=0, create_train_envs_instance=create_train_envs, device=args.device,
        create_actor_critic_instance=create_actor_critic, create_storage_instance=create_storage,
        create_algo_instance=create_algo, create_test_envs_instance=create_test_envs)

    # 7. Define learner
    learner = Learner(worker, target_steps=args.num_env_steps, log_dir=args.log_dir)

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
    parser = argparse.ArgumentParser(description='RL')

    # Environment specs
    parser.add_argument(
        '--env-id', type=str, default=None, help='Gym environment id (default None)')
    parser.add_argument(
        '--frame-skip', type=int, default=0, help='Number of frame to skip for each action (default no skip)')
    parser.add_argument(
        '--frame-stack', type=int, default=0, help='Number of frame to stack in observation (default no stack)')

    # SAC specs
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps', type=float, default=1e-8, help='Adam optimizer epsilon (default: 1e-8)')
    parser.add_argument(
        '--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--alpha', type=float, default=0.2, help='SAC alpha parameter (default: 0.2)')
    parser.add_argument(
        '--polyak', type=float, default=0.995, help='SAC polyak paramater (default: 0.995)')
    parser.add_argument(
        '--start-steps', type=int, default=1000, help='SAC num initial random steps (default: 1000)')
    parser.add_argument(
        '--buffer-size', type=int, default=10000, help='Rollouts storage size (default: 10000 transitions)')
    parser.add_argument(
        '--update-every', type=int, default=50, help='Num env collected steps between SAC network update stages (default: 50)')
    parser.add_argument(
        '--num-updates', type=int, default=50, help='Num network updates per SAC network update stage (default 50)')
    parser.add_argument(
        '--mini-batch-size', type=int, default=32, help='Mini batch size for network updates (default: 32)')
    parser.add_argument(
        '--target-update-interval', type=int, default=1, help='Num SAC network updates per target network updates (default: 1)')

    # Feature extractor model specs
    parser.add_argument(
        '--nn', default='MLP', help='Type of nn. Options are MLP, CNN, Fixup')
    parser.add_argument(
        '--restart-model', default=None, help='Restart training using the model given')
    parser.add_argument(
        '--recurrent-policy', action='store_true', default=False, help='Use a recurrent policy')
    parser.add_argument(
        '--device', default='cuda:0', help='Device to run the learner on')

    # Scheme specs
    parser.add_argument(
        '--num-env-processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')

    # General training specs
    parser.add_argument(
        '--num-env-steps', type=int, default=10e7, help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--max-time', type=int, default=-1, help='stop script after this amount of time in seconds (default: no limit)')
    parser.add_argument(
        '--log-interval', type=int, default=1, help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval', type=int, default=100, help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--log-dir', default='/tmp/ppo/', help='directory to save agent logs (default: /tmp/ppo)')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
