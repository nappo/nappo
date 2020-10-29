import time
import torch
import numpy as np
from ..base.worker import Worker as W


class CWorker(W):
    """
     Worker class handling data collection.

    This class wraps an actor_critic instance, a storage class instance and a
    train and a test vector of environments. It collects data samples, sends
    them to a central node for processing and and evaluates network versions.

    Parameters
    ----------
    index_worker : int
        Worker index.
    create_algo_instance : func
        A function that creates an algorithm class.
    create_storage_instance : func
        A function that create a rollouts storage.
    create_train_envs_instance : func
        A function to create train environments.
    create_actor_critic_instance : func
        A function that creates a policy.
    create_test_envs_instance : func
        A function to create test environments.
    initial_weights : ray object ID
        Initial model weights.

    Attributes
    ----------
    index_worker : int
        Index assigned to this worker.
    actor_critic : nn.Module
        An actor_critic class instance.
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
    ac_version : int
        Number of times the current actor critic version been has been updated.
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
                 create_algo_instance,
                 create_storage_instance,
                 create_train_envs_instance,
                 create_actor_critic_instance,
                 create_test_envs_instance=lambda x, y, c: None,
                 initial_weights=None):

        super(CWorker, self).__init__(index_worker)

        # Using Ray, worker should only see one GPU or None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Create Actor Critic instance
        self.actor_critic = create_actor_critic_instance(self.device)
        self.actor_critic.to(self.device)

        # Create Algorithm instance
        self.algo = create_algo_instance(self.actor_critic, self.device)

        # Create Storage instance and set world initial state
        self.storage = create_storage_instance(self.device)

        # Define counters and other attributes
        self.iter, self.ac_version = 0, 0
        self.update_every = self.algo.update_every or self.storage.max_size

        if initial_weights: # if remote worker

            # Create train environments, define initial train states
            self.envs_train = create_train_envs_instance(self.device, index_worker)
            self.obs, self.rhs, self.done = self.actor_critic.policy_initial_states(self.envs_train.reset())

            # Create test environments (if creation function available)
            self.envs_test = create_test_envs_instance(self.device, index_worker, mode="test")

            # Print worker information
            self.print_worker_info()

            # Set initial weights
            self.set_weights(initial_weights)

            # Collect initial samples
            print("Collecting initial samples...")
            self.collect_data(self.algo.start_steps, send=False)

    def collect_data(self, num_samples=None, send=True):
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

        if num_samples is None:
            num_samples = self.update_every

        t = time.time()
        for step in range(num_samples):

            # Predict next action, next rnn hidden state and algo-specific outputs
            act, clip_act, rhs, algo_data = self.algo.acting_step(self.obs, self.rhs, self.done)

            # Interact with envs_vector with predicted action (clipped within action space)
            obs2, reward, done, infos = self.envs_train.step(clip_act)

            # Prepare transition dict
            transition = {"obs": self.obs, "rhs": rhs, "act": act, "rew": reward, "obs2": obs2, "done": done}
            transition.update(algo_data)

            # Store transition in buffer
            self.storage.insert(transition)

            # Update current world state
            self.obs, self.rhs, self.done = obs2, rhs, done

            # Record model version used to collect data
            self.storage.ac_version = self.ac_version

        col_time = time.time() - t

        if send:

            # Get collected rollout and reset storage
            data = self.storage.get_data()
            self.storage.reset()

            # Add information to info dict
            info = {}
            info.update({"ac_version": self.ac_version})
            info.update({"scheme/seconds_to/collect": col_time})
            info.update({"collected_samples": num_samples * self.envs_train.num_envs})
            if self.iter == 0: info["collected_samples"] += (self.algo.start_steps * self.envs_train.num_envs)

            # Update counter
            self.iter += 1

            # Evaluate current network
            if self.iter % self.algo.test_every == 0:
                if self.envs_test and self.algo.num_test_episodes > 0:
                    test_perf = self.evaluate()
                    info.update({"scheme/metrics/test_performance": test_perf})

            rollouts = {"data": data, "info": info}

            return rollouts

    def evaluate(self):
        """
        Test current actor_critic version in self.envs_test.

        Returns
        -------
        mean_test_perf : float
            Average accumulated reward over all tested episodes.
        """

        completed_episodes = []
        obs = self.envs_test.reset()
        rewards = np.zeros(obs.shape[0])
        obs, rhs, done = self.actor_critic.policy_initial_states(obs)

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
        Update the worker actor_critic version with provided weights.

        weights: dict of tensors
            Dict containing actor_critic weights to be set.
        """
        self.ac_version = weights["update"]
        self.actor_critic.load_state_dict(weights["weights"])

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
