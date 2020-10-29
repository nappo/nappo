import time
import torch
import numpy as np
from ray.services import get_node_ip_address

from ..base.worker import Worker as W
from .utils import check_message, find_free_port

class CGUWorker(W):
    """
    Worker class handling remote data collection, gradient computation and
    policy updates.

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
                 initial_weights=None):

        super(CGUWorker, self).__init__(index_worker)

        # Using Ray, worker should only see one GPU or None
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Create Actor Critic instance
        self.actor_critic = create_actor_critic_instance(device)
        self.actor_critic.to(device)

        # Create Algorithm instance
        self.algo = create_algo_instance(self.actor_critic, device)

        # Define counters
        self.iter = 0

        if initial_weights: # if remote worker

            # Create train environments, define initial train states
            self.envs_train = create_train_envs_instance(device, index_worker)
            self.obs, self.rhs, self.done = self.actor_critic.policy_initial_states(self.envs_train.reset())

            # Create test environments (if creation function available)
            self.envs_test = create_test_envs_instance(device, index_worker, mode="test")

            # Create Storage instance and set world initial state
            self.storage = create_storage_instance(device)
            self.update_every = self.algo.update_every or self.storage.max_size

            # Print worker information
            self.print_worker_info()
            self.set_weights(initial_weights)

            # Collect initial samples
            print("Collecting initial samples...")
            self.collect_data(self.algo.start_steps, self.algo.start_steps)

    def collect_data(self, num_steps=None, min_fraction=1.0):
        """
        Collect data from interactions with the environments.

        Parameters
        ----------
        num_steps : int
            Target number of train environment steps to take.
        min_steps : int
            Minimum number of train environment steps to stop at if other
            workers have finished their collection task.
        """

        if self.iter % (self.algo.num_epochs * self.algo.num_mini_batch) != 0:
            return

        if num_steps is None:
            num_steps = int(self.update_every)

        min_steps = int(num_steps * min_fraction)

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
            self.storage.ac_version = self.iter

            if check_message("sample") == b"stop" and step >= min_steps:
                self.num_steps_reached = step
                break

        # Record and return metrics
        self.collect_time = time.time() - t
        self.collect_samples = num_steps * self.envs_train.num_envs

    def compute_gradients(self, batch):
        """
        Calculate actor critic gradients and average them with grads from
        other workers.

        Parameters
        ----------
        batch : dict
            data batch containing all required tensors to compute algo loss.

        Returns
        -------
        info : dict
            Summary dict of relevant gradient-related information.
        """

        t = time.time()
        grads, info = self.algo.compute_gradients(batch, grads_to_cpu=False)
        compute_grads_t = time.time() - t

        ###### ALLREDUCE ######################################################

        t = time.time()
        if torch.cuda.is_available():
            for g in grads:
                torch.distributed.all_reduce(g, op=torch.distributed.ReduceOp.SUM)
        else:
            torch.distributed.all_reduce_coalesced(grads, op=torch.distributed.ReduceOp.SUM)

        for p in self.actor_critic.parameters():
            if p.grad is not None:
                p.grad /= self.distributed_world_size
        avg_grads_t = time.time() - t

        #######################################################################

        info.update({"scheme/seconds_to/compute_grads": compute_grads_t})
        info.update({"scheme/seconds_to/avg_grads_t": avg_grads_t})

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
            act, clip_act, rhs, _ = self.algo.acting_step(obs, rhs, done, deterministic=True)

            # Interact with env with predicted action (clipped within action space)
            obs2, reward, done, _ = self.envs_test.step(clip_act)

            # Keep track of episode rewards and completed episodes
            rewards += reward.cpu().squeeze(-1).numpy()
            completed_episodes.extend(
                rewards[done.cpu().squeeze(-1).numpy() == 1.0].tolist())
            rewards[done.cpu().squeeze(-1).numpy() == 1.0] = 0.0

            obs = obs2

        return np.mean(completed_episodes)

    def step(self):
        """
        Perform logical learning step. Training proceeds interleaving collection
        of data samples with gradient computations and policy updates.

        Returns
        -------
        info : dict
            Summary dict with relevant step information.

        """

        # Collect data and prepare data batches
        if self.iter % (self.algo.num_epochs * self.algo.num_mini_batch) == 0:
            self.storage.before_update(self.actor_critic, self.algo)
            self.batches = self.storage.generate_batches(
                self.algo.num_mini_batch, self.algo.mini_batch_size,
                self.algo.num_epochs, self.actor_critic.is_recurrent)
            collect_time, collected_samples = self.collect_time, self.collect_samples
        else:
            collect_time, collected_samples = 0.0, 0

        # Get next batch
        batch = self.batches.__next__()

        # Compute gradients
        info = self.compute_gradients(batch)

        # Update model
        self.update_networks()

        # Add extra information to info dict
        info.update({"ac_update_num": self.iter})
        info.update({"collected_samples": collected_samples})
        info.update({"scheme/seconds_to/collect": collect_time})
        info.update({"scheme/metrics/collection_gradient_delay": self.iter - self.storage.ac_version})
        if self.iter == 0: info["collected_samples"] += self.algo.start_steps * self.envs_train.num_envs

        # Update counter
        self.iter += 1

        # Evaluate current network
        if self.iter % self.algo.test_every == 0:
            if self.envs_test and self.algo.num_test_episodes > 0:
                test_perf = self.evaluate()
                info.update({"scheme/metrics/test_performance": test_perf})

        return info

    def setup_torch_data_parallel(self, url, world_rank, world_size, backend):
        """Join a torch process group for distributed SGD."""
        torch.distributed.init_process_group(
            backend=backend,
            init_method=url,
            rank=world_rank,
            world_size=world_size)
        self.distributed_world_size = world_size

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return get_node_ip_address()

    def find_free_port(self):
        """Returns a free port on the current node."""
        return find_free_port()

    def set_weights(self, weights):
        """
        Update the worker actor_critic version with provided weights.

        weights: dict of tensors
            Dict containing actor_critic weights to be set.
        """
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