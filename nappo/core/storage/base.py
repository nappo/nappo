import torch
from abc import ABC, abstractmethod


class Storage:
    """Base class for all storage components"""

    @classmethod
    @abstractmethod
    def storage_factory(cls):
        """Returns a function to create new Storage instances"""
        raise NotImplementedError

    on_policy_data_fields = ("obs", "obs2", "rhs", "act", "rew", "val", "logp", "done")

    def __init__(self, size, device=torch.device("cpu")):

        self.device = device
        self.max_size = size
        self.reset()

    def init_tensors(self, sample):
        """
        Lazy initialization of data tensors from a sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        raise NotImplementedError


    def get_data(self):
        """Return currently stored data."""
        return {k: v.cpu() for k, v in self.data.items() if v is not None}

    def add_data(self, new_data):
        """
        Replace currently stored data.

        Parameters
        ----------
        new_data : dict
            Dictionary of env transition samples to replace self.data with.
        """
        self.data = {k: v.to(self.device) for k, v in new_data.items()}

    def reset(self):
        """Set class counters to zero and remove stored data"""
        self.step, self.size = 0, 0
        self.data = {k: None for k in self.on_policy_data_fields} # lazy init

    def insert(self, sample):
        """
        Store new transition sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        raise NotImplementedError

    def before_update(self, actor_critic, algo):
        """
        Before updating actor policy model, compute returns and advantages.

        Parameters
        ----------
        actor_critic : ActorCritic
            An actor_critic class instance.
        algo : an algorithm class
            An algorithm class instance.
        """
        raise NotImplementedError

    def after_update(self):
        """After updating actor policy model, make sure self.step is at 0."""
        raise NotImplementedError

    def generate_batches(self, num_mini_batch, mini_batch_size, num_epochs=1, recurrent_ac=False, shuffle=True):
        """
        Returns a batch iterator to update actor critic.

        Parameters
        ----------
        num_mini_batch : int
           Number mini batches per epoch.
        mini_batch_size : int
            Number of samples contained in each mini batch.
        num_epochs : int
            Number of epochs.
        recurrent_ac : bool
            Whether actor critic policy is a RNN or not.
        shuffle : bool
            Whether to shuffle collected data or generate sequential

        Yields
        ______
        batch : dict
            Generated data batches.
        """
        raise NotImplementedError
