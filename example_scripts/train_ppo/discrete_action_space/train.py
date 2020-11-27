import os
import ray
import sys
import time
import json
import argparse

from nappo import utils
from nappo import Learner
from nappo.schemes import Scheme
from nappo.core.algos import PPO
from nappo.core.envs import VecEnv
from nappo.core.storages import OnPolicyGAEBuffer
from nappo.core.actors import OnPolicyActorCritic, get_feature_extractor
from nappo.envs import make_atari_train_env, make_atari_test_env


def main():

    args = get_args()
    utils.cleanup_log_dir(args.log_dir)
    args_dict = vars(args)
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
    train_envs_factory, action_space, obs_space = VecEnv.create_factory(
        env_fn=make_atari_train_env,
        env_kwargs={"env_id": args.env_id, "frame_stack": args.frame_stack},
        vec_env_size=args.num_env_processes, log_dir=args.log_dir,
        info_keywords=('rr', 'rrr', 'lives'))

    # 2. Define Test Vector of Envs (Optional)
    test_envs_factory, _, _ = VecEnv.create_factory(
        env_fn=make_atari_test_env,
        env_kwargs={"env_id": args.env_id, "frame_stack": args.frame_stack},
        vec_env_size=args.num_env_processes, log_dir=args.log_dir)

    # 3. Define RL training algorithm
    algo_factory = PPO.create_factory(
        lr=args.lr, eps=args.eps, num_epochs=args.ppo_epoch, clip_param=args.clip_param,
        entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm, num_mini_batch=args.num_mini_batch,
        use_clipped_value_loss=args.use_clipped_value_loss, gamma=args.gamma)

    # 4. Define RL Policy
    actor_factory = OnPolicyActorCritic.create_factory(
        obs_space, action_space,
        feature_extractor_network=get_feature_extractor(args.nn),
        recurrent_policy=args.recurrent_policy,
        restart_model=args.restart_model)

    # 5. Define rollouts storage
    storage_factory = OnPolicyGAEBuffer.create_factory(size=args.num_steps, gae_lambda=args.gae_lambda)

    # 6. Define scheme
    params = {}

    # add core modules
    params.update({
        "algo_factory": algo_factory,
        "actor_factory": actor_factory,
        "storage_factory": storage_factory,
        "train_envs_factory": train_envs_factory,
        "test_envs_factory": test_envs_factory,
    })

    # add collection specs
    params.update({
        "col_remote_workers": 0,
        "col_communication": "synchronous",
        "col_worker_resources": {"num_cpus": 1, "num_gpus": 0.125},
        "sync_col_specs": {"fraction_samples": 1.0, "fraction_workers": 1.0}
    })

    # add gradient specs
    params.update({
        "grad_remote_workers": 4,
        "grad_communication": "synchronous",
        "grad_worker_resources": {"num_cpus": 1, "num_gpus": 0.125},
    })

    # add update specs
    params.update({
        "local_device": None,
        "update_execution": "decentralised",
    })

    scheme = Scheme(**params)

    # 7. Define learner
    learner = Learner(scheme, target_steps=args.num_env_steps, log_dir=args.log_dir)

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
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action (default no skip)')
    parser.add_argument(
        '--frame-stack', type=int, default=0,
        help='Number of frame to stack in observation (default no stack)')

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
        '--nn', default='CNN', help='Type of nn. Options are MLP, CNN, Fixup')
    parser.add_argument(
        '--restart-model', default=None,
        help='Restart training using the model given')
    parser.add_argument(
        '--recurrent-policy', action='store_true', default=False,
        help='Use a recurrent policy')

    # Scheme specs
    parser.add_argument(
        '--scheme', default='3cs',
        help='Distributed training scheme name (default: 3cs)')
    parser.add_argument(
        '--num-env-processes', type=int, default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-workers', type=int, default=1, help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--cluster', action='store_true', default=False, help='script is running in a cluster')

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
