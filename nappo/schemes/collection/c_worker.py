import ray
import time
import torch
import numpy as np
from collections import deque
from ..base.worker import Worker as W
from ..utils import check_message


class CWorker(W):
    """
     Worker class handling data collection.

    This class wraps an actor instance, a storage class instance and a
    train and a test vector of environments. It collects data samples, sends
    them to a central node for processing and and evaluates network versions.

    Parameters
    ----------
    index_worker : int
        Worker index.
    algo_factory : func
        A function that creates an algorithm class.
    storage_factory : func
        A function that create a rollouts storage.
    train_envs_factory : func
        A function to create train environments.
    actor_factory : func
        A function that creates a policy.
    test_envs_factory : func
        A function to create test environments.
    initial_weights : ray object ID
        Initial model weights.

    Attributes
    ----------
    index_worker : int
        Index assigned to this worker.
    actor : nn.Module
        An actor class instance.
    algo : an algorithm class
        An algorithm class instance.
    envs_train : VecEnv
        A VecEnv class instance with the train environments.
    envs_test : VecEnv
        A VecEnv class instance with the test environments.
    storage : a rollout storage class
        A Storage class instance.
    iter : int
         Number of times gradients have been computed and sent.
    actor_version : int
        Number of times the current actor version been has been updated.
    update_every : int
        Number of data samples to collect between network update stages.
    obs : torch.tensor
        Latest train environment observation.
    rhs : torch.tensor
        Latest policy recurrent hidden state.
    done : torch.tensor
        Latest train environment done flag.
    """

    def __init__(self,
                 index_worker,
                 algo_factory,
                 actor_factory,
                 storage_factory,
                 train_envs_factory=lambda x, y, z: None,
                 test_envs_factory=lambda x, y, z: None,
                 initial_weights=None,
                 device=None):

        self.index_worker = index_worker
        super(CWorker, self).__init__(index_worker)

        # Computation device
        dev = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        # Create Actor Critic instance
        self.actor = actor_factory(self.device)
        self.actor.to(self.device)

        # Create Algorithm instance
        self.algo = algo_factory(self.actor, self.device)

        # Create Storage instance and set world initial state
        self.storage = storage_factory(self.device)

        # Define counters and other attributes
        self.iter, self.actor_version, self.samples_collected = 0, 0, 0
        self.update_every = self.algo.update_every or self.storage.max_size

        # Create train environments, define initial train states
        self.envs_train = train_envs_factory(self.device, index_worker, mode="train")
        self.obs, self.rhs, self.done = self.actor.policy_initial_states(self.envs_train.reset())

        # Create test environments (if creation function available)
        self.envs_test = test_envs_factory(self.device, index_worker, mode="test")

        if initial_weights: # if remote worker

            # Set initial weights
            self.set_weights(initial_weights)

            # Print worker information
            self.print_worker_info()

        if self.envs_train:

            # Define train performance tracking variables
            self.train_perf = deque(maxlen=100)
            self.acc_reward = torch.zeros_like(self.done)

            # Collect initial samples
            print("Collecting initial samples...")
            self.collect_data(self.algo.start_steps)


    def collect_data(self, min_fraction=1.0):
        """ _ """

        # Collect train data
        col_time, train_perf = self.collect_train_data(min_fraction=min_fraction)

        # Get collected rollout and reset storage
        data = self.storage.get_data()
        self.storage.reset()

        # Add information to info dict
        info = {}
        info.update({"time/collect": col_time})
        info.update({"performance/train": train_perf})
        info.update({"col_version": self.actor_version})
        info.update({"collected_samples": self.samples_collected})
        self.samples_collected = 0

        # Evaluate current network on test environments
        if self.iter % self.algo.test_every == 0:
            if self.envs_test and self.algo.num_test_episodes > 0:
                test_perf = self.evaluate()
                info.update({"performance/test": test_perf})

        # Update counter
        self.iter += 1

        # Return data
        rollouts = {"data": data, "info": info}

        return rollouts

    def collect_train_data(self, num_steps=None, min_fraction=1.0):
        """
        Collect data from interactions with the environments.

        Parameters
        ----------
        num_steps : int
            Target number of train environment steps to take.
        send : bool
            If true, this function returns the collected rollouts.

        Returns
        -------
        rollouts : dict
            Dict containing collected data and ohter relevant information
            related to the collection process.
        """
        t = time.time()
        num_steps = num_steps or int(self.update_every)
        min_steps = int(num_steps * min_fraction)

        for step in range(num_steps):

            # Predict next action, next rnn hidden state and algo-specific outputs
            act, clip_act, rhs, algo_data = self.algo.acting_step(self.obs, self.rhs, self.done)

            # Interact with envs_vector with predicted action (clipped within action space)
            obs2, reward, done, infos = self.envs_train.step(clip_act)

            # Handle end of episode
            self.acc_reward += reward
            self.train_perf.extend(self.acc_reward[done == 1.0].tolist())
            self.acc_reward[done == 1.0] = 0.0

            # Prepare transition dict
            transition = {"obs": self.obs, "rhs": rhs, "act": act, "rew": reward, "obs2": obs2, "done": done}
            transition.update(algo_data)

            # Store transition in buffer
            self.storage.insert(transition)

            # Update current world state
            self.obs, self.rhs, self.done = obs2, rhs, done

            # Keep track of num collected samples
            self.samples_collected += self.envs_train.num_envs

            if self.index_worker > 0: # Only remote workers
                # Check stop message for the synchronised case
                if check_message("sample") == b"stop" and step >= min_steps:
                    break

        col_time = time.time() - t
        train_perf = 0 if len(self.train_perf) == 0 else sum(self.train_perf) / len(self.train_perf)

        return col_time, train_perf

    def evaluate(self):
        """
        Test current actor version in self.envs_test.

        Returns
        -------
        mean_test_perf : float
            Average accumulated reward over all tested episodes.
        """

        completed_episodes = []
        obs = self.envs_test.reset()
        rewards = np.zeros(obs.shape[0])
        obs, rhs, done = self.actor.policy_initial_states(obs)

        while len(completed_episodes) < self.algo.num_test_episodes:

            # Predict next action and rnn hidden state
            act, clip_act, rhs, _ = self.algo.acting_step(obs, rhs, done, deterministic=True)

            # Interact with env with predicted action (clipped within action space)
            obs2, reward, done, _ = self.envs_test.step(clip_act)

            # Keep track of episode rewards and completed episodes
            rewards += reward.cpu().squeeze(-1).numpy()
            completed_episodes.extend(rewards[done.cpu().squeeze(-1).numpy() == 1.0].tolist())
            rewards[done.cpu().squeeze(-1).numpy() == 1.0] = 0.0

            obs = obs2

        return np.mean(completed_episodes)

    def set_weights(self, weights):
        """
        Update the worker actor version with provided weights.

        weights: dict of tensors
            Dict containing actor weights to be set.
        """
        self.actor_version = weights["version"]
        self.actor.load_state_dict(weights["weights"])

    def update_algo_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of Worker.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        """
        self.algo.update_algo_parameter(parameter_name, new_parameter_value)