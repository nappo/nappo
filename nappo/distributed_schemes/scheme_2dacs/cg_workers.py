import time
import torch
import numpy as np
from ..base.worker import Worker as W
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config


class CGWorker(W):
    """
     Worker class handling data collection and gradient computation.

    This class wraps an actor instance, a storage class instance and a
    train and a test vector of environments. It collects data, computes gradients
    and evaluates network versions following a logic defined in function self.step(),
    which will be called from the Learner class.

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
        Times gradients have been computed and sent.
    ac_version : int
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
                 train_envs_factory,
                 test_envs_factory=lambda x, y, c: None,
                 initial_weights=None):

        super(CGWorker, self).__init__(index_worker)

        # Using Ray, worker should only see one GPU or None
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Create Actor Critic instance
        self.actor = actor_factory(device)
        self.actor.to(device)

        # Create Algorithm instance
        self.algo = algo_factory(self.actor, device)

        # Define counters and other attributes
        self.iter = 0
        self.ac_version = 0
        self.num_updates = self.algo.num_epochs * self.algo.num_mini_batch

        if initial_weights: # if remote worker

            # Create train environments, define initial train states
            self.envs_train = train_envs_factory(device, index_worker)
            self.obs, self.rhs, self.done = self.actor.policy_initial_states(self.envs_train.reset())

            # Create test environments (if creation function available)
            self.envs_test = test_envs_factory(device, index_worker, mode="test")

            # Create Storage instance and set world initial state
            self.storage = storage_factory(device)
            self.update_every = self.algo.update_every or self.storage.max_size

            # Print worker information
            self.print_worker_info()

            # Set initial weights
            self.set_weights(initial_weights)

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

        # Record and return metrics
        total_time = time.time() - t
        total_samples = num_steps * self.envs_train.num_envs

        return total_time, total_samples

    def compute_gradients(self, batch):
        """
        Calculate actor gradients.

        Parameters
        ----------
        batch : dict
            data batch containing all required tensors to compute algo loss.

        Returns
        -------
        grads: list of tensors
            List of actor gradients.
        info : dict
            Summary dict of relevant gradient-related information.
        """

        t = time.time()
        grads, info = self.algo.compute_gradients(batch)
        info.update({"scheme/seconds_to/compute_grads": time.time() - t})

        return grads, info

    def update_networks(self, gradients):
        """Update Actor Critic model"""
        self.algo.apply_gradients(gradients)

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

    def step(self):
        """
        Perform logical learning step. Training proceeds interleaving collection
        of data samples with policy gradient computations.

        Returns
        -------
        grads: list of tensors
            List of actor gradients.
        info : dict
            Summary dict with relevant step information.
        """

        # Collect data and prepare data batches
        if self.iter % self.num_updates == 0:
            collect_time, collected_samples = self.collect_data(self.update_every)
            self.storage.before_update(self.actor, self.algo)
            self.batches = self.storage.generate_batches(
                self.algo.num_mini_batch, self.algo.mini_batch_size,
                self.algo.num_epochs, self.actor.is_recurrent)
        else:
            collect_time, collected_samples = 0.0, 0

        # Compute gradients, get algo info
        grads, info = self.compute_gradients(self.batches.__next__())

        # Add extra information to info dict
        info.update({"ac_update_num": self.iter})
        info.update({"collected_samples": collected_samples})
        info.update({"scheme/seconds_to/collect": collect_time})
        info.update({"scheme/metrics/collection_gradient_delay": self.iter - self.storage.ac_version})
        if self.iter == 0: info["collected_samples"] += self.start_steps

        # Evaluate current network
        if self.iter % self.algo.test_every == 0:
            if self.envs_test and self.algo.num_test_episodes > 0:
                test_perf = self.evaluate()
                info.update({"scheme/metrics/test_performance": test_perf})

        # Update counter
        self.iter += 1
        
        return grads, info

    def set_weights(self, weights):
        """
        Update the worker actor version with provided weights.

        weights: dict of tensors
            Dict containing actor weights to be set.
        """
        self.ac_version = weights["update"]
        self.algo.set_weights(weights["weights"])

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

class CGWorkerSet(WS):
    """
    Class to better handle the operations of ensembles of CGWorkers.

    Parameters
    ----------
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
    worker_remote_config : dict
        Ray resource specs for the remote workers.
    num_workers : int
        Number of remote workers in the worker set.

    Attributes
    ----------
    worker_class : python class
        Worker class to be instantiated to create Ray remote actors.
    remote_config : dict
        Ray resource specs for the remote workers.
    worker_params : dict
        Keyword arguments of the worker_class.
    num_workers : int
        Number of remote workers in the worker set.
    """

    def __init__(self,
                 algo_factory,
                 actor_factory,
                 storage_factory,
                 train_envs_factory,
                 test_envs_factory=lambda x, y, c: None,
                 worker_remote_config=default_remote_config,
                 num_workers=1):

        self.worker_class = CGWorker
        default_remote_config.update(worker_remote_config)
        self.remote_config = default_remote_config
        self.worker_params = {
            "algo_factory": algo_factory,
            "storage_factory": storage_factory,
            "test_envs_factory": test_envs_factory,
            "train_envs_factory": train_envs_factory,
            "actor_factory": actor_factory,
        }

        super(CGWorkerSet, self).__init__(
            worker=self.worker_class,
            worker_params=self.worker_params,
            worker_remote_config=self.remote_config,
            num_workers=num_workers)