import os
import time
import torch
import numpy as np
from shutil import copy2


class CGUWorker:
    """
    Worker class handling data collection, gradient computation and policy updates.

    This class wraps an actor_critic instance, a storage class instance and a
    train and a test vector of environments. It collects data, computes gradients,
    updates the networks and evaluates network versions following a logic
    defined in function self.step(), which will be called from the Learner.

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
    device : torch.device
        CPU or specific GPU to use for computation.

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
    num_updates : int
        Times actor critic has been updated.
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
                 device="cuda:0"):

        self.index_worker = index_worker
        device = torch.device(device)

        # Create Actor Critic instance
        self.actor_critic = create_actor_critic_instance(device)
        self.actor_critic.to(device)

        # Create Algorithm instance
        self.algo = create_algo_instance(self.actor_critic, device)

        # Create train environments, define initial train states
        self.envs_train = create_train_envs_instance(device, index_worker)
        self.obs, self.rhs, self.done = self.actor_critic.policy_initial_states(self.envs_train.reset())

        # Create test environments (if creation function available)
        self.envs_test = create_test_envs_instance(device, index_worker, mode="test")

        # Create Storage instance
        self.storage = create_storage_instance(device)

        # Define counters and other attributes
        self.num_updates = 0  # times step has been called
        self.update_every = self.algo.update_every or self.storage.max_size

        # Collect initial samples
        print("Collecting initial samples...")
        _, self.start_steps = self.collect_data(self.algo.start_steps)

    def collect_data(self, num_steps):
        """
        Collect data from interactions with the environments.

        Parameters
        ----------
        num_steps : int
            Target number of train environment steps to take.

        Returns
        -------
        total_time : float
            Elapsed time in seconds.
        total_samples : int
            Total num of collected samples.
        """

        t = time.time()
        for step in range(num_steps):

            # Predict next action, next rnn hidden state and algo-specific outputs
            act, clip_act, rhs, algo_data = self.algo.acting_step(self.obs, self.rhs, self.done)

            # Interact with env with predicted action (clipped within action space)
            obs2, reward, done, infos = self.envs_train.step(clip_act)

            # Prepare transition dict
            transition = {"obs": self.obs, "rhs": rhs,"act": act, "rew": reward, "obs2": obs2, "done": done}
            transition.update(algo_data)

            # Store transition in buffer
            self.storage.insert(transition)

            # Update current world state
            self.obs, self.rhs, self.done = obs2, rhs, done

        # Record model version used to collect data
        self.storage.ac_version = self.num_updates

        # Record and return metrics
        total_time = time.time() - t
        total_samples = num_steps * self.envs_train.num_envs

        return total_time, total_samples

    def compute_gradients(self, batch):
        """
        Calculate actor critic gradients.

        Parameters
        ----------
        batch : dict
            data batch containing all required tensors to compute algo loss.

        Returns
        -------
        info : dict
            Summary dict with relevant gradient-related information.
        """

        t = time.time()
        _, info = self.algo.compute_gradients(batch)
        info.update({"scheme/seconds_to/compute_grads": time.time() - t})

        return info

    def update_networks(self):
        """Update Actor Critic model"""
        self.algo.apply_gradients()

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
            act, clip_act, rhs, _ = self.algo.acting_step(obs, rhs, done,
                                                          deterministic=True)

            # Interact with env with predicted action (clipped within action space)
            obs2, reward, done, _ = self.envs_test.step(clip_act)

            # Keep track of episode rewards and completed episodes
            rewards += reward.cpu().squeeze(-1).numpy()
            completed_episodes.extend(
                rewards[done.cpu().squeeze(-1).numpy() == 1.0].tolist())
            rewards[done.cpu().squeeze(-1).numpy() == 1.0] = 0.0

            obs = obs2

        mean_test_perf = np.mean(completed_episodes)

        return mean_test_perf

    def step(self):
        """
        Perform logical learning step. Training proceeds interleaving collection
        of data samples with policy updates.

        Returns
        -------
        info : dict
            Summary dict of relevant step information.
        """

        # Collect data and prepare data batches
        if self.num_updates % (self.algo.num_epochs * self.algo.num_mini_batch) == 0:
            collect_time, collected_samples = self.collect_data(self.update_every)
            self.storage.before_update(self.actor_critic, self.algo)
            self.batches = self.storage.generate_batches(
                self.algo.num_mini_batch, self.algo.mini_batch_size,
                self.algo.num_epochs, self.actor_critic.is_recurrent)
        else:
            collect_time, collected_samples = 0.0, 0

        # Get next batch
        batch = self.batches.__next__()

        # Compute gradients
        info = self.compute_gradients(batch)

        # Update model
        self.update_networks()

        # Add extra information to info dict
        info.update({"ac_update_num": self.num_updates})
        info.update({"collected_samples": collected_samples})
        info.update({"scheme/seconds_to/collect": collect_time})
        info.update({"scheme/metrics/collection_gradient_delay": self.num_updates - self.storage.ac_version})
        if self.num_updates == 0: info["collected_samples"] += self.start_steps

        # Update counter
        self.num_updates += 1

        # Evaluate current network
        if self.num_updates % self.algo.test_every == 0:
            if self.envs_test and self.algo.num_test_episodes > 0:
                test_perf = self.evaluate()
                info.update({"scheme/metrics/test_performance": test_perf})

        return info

    def save_model(self, fname):
        """
        Save current version of actor_critic as a torch loadable checkpoint.

        Parameters
        ----------
        fname : str
            Filename given to the checkpoint.

        Returns
        -------
        save_name : str
            Path to saved file.
        """
        torch.save(self.algo.actor_critic.state_dict(), fname + ".tmp")
        os.rename(fname + '.tmp', fname)
        save_name = fname + ".{}".format(self.num_updates)
        copy2(fname, save_name)
        return save_name

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

    def stop(self):
        """Execute required code to end actor critic updates"""
        pass



