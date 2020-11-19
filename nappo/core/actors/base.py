import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .neural_networks.feature_extractors import get_model


class Actor(nn.Module, ABC):
    """
    Actor critic class for Off-Policy algorithms.

    It contains a policy network (actor) to predict next actions and one or two
    Q networks.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    feature_extractor_network : nn.Module
        PyTorch nn.Module to extract features in all networks.
    feature_extractor_kwargs : dict
        Keyword arguments for the feature extractor network.
    recurrent_policy : bool
        Whether to use a RNN as a policy.
    recurrent_hidden_size : int
        Policy RNN hidden size.
    create_double_q_critic : bool
        Whether to instantiate a second Q network or not.

    Attributes
    ----------
    policy_net : nn.module
        Neural network that extracts features from the input env obs.
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    dist : nn.module
        Neural network that predict a prob distribution over the action space
    q1 : nn.module
        Neural network that predicts a Q-value for a given env obs and action.
    q2 : nn.module
        A second neural network to predict a Q-value for a given env obs and action.
    scale : nn.module
        Maps actions from [space.low, space.high] range to [-1, 1] range.
    unscale : nn.module
        Maps actions from [-1, 1] range to [space.low, space.high] range.

    Examples
    --------
    """
    def __init__(self,
                 input_space,
                 action_space,
                 feature_extractor_network=get_model("MLP"),
                 feature_extractor_kwargs={},
                 recurrent_policy=False,
                 recurrent_hidden_size=512,
                 *args):

        super(Actor, self).__init__()
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            restart_model=None,
            recurrent_policy=False,
            recurrent_hidden_size=512,
            feature_extractor_kwargs={},
            feature_extractor_network=get_model("MLP"),
            *args):
        """
        Returns a function that creates actor critic instances.

        Parameters
        ----------
        input_space : gym.Space
            Environment observation space.
        action_space : gym.Space
            Environment action space.
        restart_model : str
            Path to a previously trained ActorCritic checkpoint to be loaded.
        feature_extractor_network : nn.Module
            PyTorch nn.Module used as the features extraction block in all networks.
        feature_extractor_kwargs : dict
            Keyword arguments for the feature extractor network.
        recurrent_policy : bool
            Whether to use a RNN as a policy.
        recurrent_hidden_size : int
            Policy RNN hidden size.
        create_double_q_critic : bool
            whether to instantiate a second Q network or not.

        Returns
        -------
        create_actor_critic_instance : func
            creates a new OffPolicyActorCritic class instance.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_recurrent(self, *args):
        """Returns True if the policy network has recurrency."""
        raise NotImplementedError

    @property
    @abstractmethod
    def recurrent_hidden_state_size(self):
        """Size of policy recurrent hidden state"""
        raise NotImplementedError

    @abstractmethod
    def policy_initial_states(self, obs, *args):
        """
        Returns all policy inputs to predict the environment initial action.

        Parameters
        ----------
        obs : torch.tensor
            Initial environment observation.

        Returns
        -------
        obs : torch.tensor
            Initial environment observation.
        rhs : torch.tensor
            Initial recurrent hidden state (will contain zeroes).
        done : torch.tensor
            Initial done tensor, indicating the environment is not done.
        """
        raise NotImplementedError

    @abstractmethod
    def get_action(self, obs, rhs, dones, deterministic=False, *args):
        """
        Predict and return next action, along with other information.

        Parameters
        ----------
        obs : torch.tensor
            Current environment observation.
        rhs : torch.tensor
            Current recurrent hidden state.
        done : torch.tensor
            Current done tensor, indicating if episode has finished.
        deterministic : bool
            Whether to randomly sample action from predicted distribution or take the mode.

        Returns
        -------
        action : torch.tensor
            Next action sampled.
        clipped_action : torch.tensor
            Next action sampled, but clipped to be within the env action space.
        logp_action : torch.tensor
            Log probability of `action` within the predicted action distribution.
        rhs : torch.tensor
            Updated recurrent hidden state.
        entropy_dist : torch.tensor
            Entropy of the predicted action distribution.
        """
        raise NotImplementedError
