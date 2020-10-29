""" Neural Network Architectures. """

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init

# ---- Neural Network Base Class ----------------------------------------------

class NNBase(nn.Module):
    """
    Base Neural Network class.

    Parameters
    ----------
    input_shapes : list of tuples
        List of all input shapes.
    output_shape : tuple
        Shape of output feature map.
    feature_extractor : nn.Module
        PyTorch nn.Module used as the features extraction block.
    feature_extractor_kwargs : dict
        Keyword arguments for the feature extractor network.
    recurrent : bool
        Whether to use recurrency or not.
    recurrent_input_size (int):
        Input size of GRU layer.
    recurrent_hidden_size (int):
        Hidden size of GRU layer.
    activation : func
        Non-linear activation function.
    final_activation : bool
        Whether or not to apply nonlinearity after last layer.

    Attributes
    ----------
    is_recurrent : bool
        True if this is a recurrent neural network.
    num_outputs : int
        Output feature map size (as in np.prod(self.output_shape)).
    output_shape : tuple
        Output feature map shape.
    recurrent_hidden_state_size : int
        Recurrent hidden state size
    feature_extractor : nn.Module
        Neural network feature extractor block.
    gru : nn.Module
        Neural network recurrency block.
    output : nn.Module
        Neural network final layer.
    """
    def __init__(self,
                 input_shapes,
                 feature_extractor,
                 feature_extractor_kwargs,
                 recurrent=False,
                 output_shape=(256,),
                 recurrent_hidden_size=512,
                 activation=nn.ReLU,
                 final_activation=False):

        super(NNBase, self).__init__()

        self.output_shape = output_shape
        self._num_outputs = np.prod(output_shape)
        self._recurrent = recurrent
        self._recurrent_hidden_size = recurrent_hidden_size

        if len(input_shapes) == 1: obs_shape, action_shape = input_shapes[0], 0
        elif len(input_shapes) == 2: obs_shape, action_shape = input_shapes
        else: raise NotImplementedError

        self.feature_extractor = feature_extractor(obs_shape, **feature_extractor_kwargs)
        feature_map_size = int(np.prod(self.feature_extractor(torch.randn(1, *obs_shape)).shape))

        if recurrent:
            self.gru = nn.GRU(feature_map_size, recurrent_hidden_size)
            feature_map_size = recurrent_hidden_size

        output = [nn.Linear(feature_map_size + np.prod(action_shape), self._num_outputs)]
        if final_activation: output += [activation()]
        self.output = nn.Sequential(*output)

        self.train()

    @property
    def is_recurrent(self):
        """Returns True if this is a recurrent neural network"""
        return self._recurrent

    @property
    def num_outputs(self):
        """Returns output feature map size"""
        return self._num_outputs

    @property
    def recurrent_hidden_state_size(self):
        """Returns recurrent hidden state size"""
        return self._recurrent_hidden_size

    def initial_states(self, obs):
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
        done = torch.zeros(obs.size(0), 1).to(obs.device)
        rhs = torch.zeros(obs.size(0), self._recurrent_hidden_size).to(obs.device)
        return obs, rhs, done

    def _forward_gru(self, x, hxs, done):
        """
        Fast forward pass GRU network.
        from Ilya Kostrikov.PyTorch Implementations of Reinforcement Learning Algorithms.
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail. 2018

        Parameters
        ----------
        x : torch.tensor
            Feature map obtained from environment observation.
        hxs : torch.tensor
            Current recurrent hidden state.
        done : torch.tensor
            Current done tensor, indicating if episode has finished.

        Returns
        -------
        x : torch.tensor
            Feature map obtained after GRU.
        hxs : torch.tensor
            Updated recurrent hidden state.
        """

        masks = 1 - done

        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

    def forward(self, inputs, hxs=None, done=None):
        """
        Forward pass Neural Network

        Parameters
        ----------
        inputs : torch.tensor or tuple
            Input data, containing obs or obs + action.
        hxs : torch.tensor
            Current recurrent hidden state.
        done : torch.tensor
            Current done tensor, indicating if episode has finished.

        Returns
        -------
        out : torch.tensor
            Output feature map.
        hxs : torch.tensor
            Updated recurrent hidden state.
        """

        if isinstance(inputs, tuple):
            obs, act = inputs
        else:
            obs, act = inputs, None

        x = self.feature_extractor(obs)
        x = x.view(x.size(0), -1)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, hxs, done)

        if act is not None:
            x = torch.cat([x, act], dim=-1)

        out = self.output(x)
        assert tuple(out.shape[1:]) == self.output_shape

        return out, hxs

# ---- Multilayer Perceptron --------------------------------------------------

class MLP(nn.Module):
    """
    Multilayer Perceptron network

    Parameters
    ----------
    input_shape : tuple
        Shape input tensors.
    hidden_sizes : list
        Hidden layers sizes.
    activation : func
        Non-linear activation function.

    Attributes
    ----------
    feature_extractor : nn.Module
        Neural network feature extractor block.
    """
    def __init__(self, input_shape, hidden_sizes=[256, 256], activation=nn.ReLU):
        super(MLP, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init. constant_(x, 0), nn.init.calculate_gain('relu'))

        # Define feature extractor
        layers = []
        sizes = [input_shape[0]] + hidden_sizes
        for j in range(len(sizes) - 1):
            layers += [init_(nn.Linear(sizes[j], sizes[j + 1])), activation()]
        self.feature_extractor = nn.Sequential(*layers)

        self.train()


    def forward(self, inputs):
        """
        Forward pass Neural Network

        Parameters
        ----------
        inputs : torch.tensor
            Input data.

        Returns
        -------
        out : torch.tensor
            Output feature map.
        """
        out = self.feature_extractor(inputs)
        return out

# ---- Basic Convolutional Neural Network -------------------------------------

class CNN(nn.Module):
    """
    Convolutional Neural Network.

    Parameters
    ----------
    input_shape : tuple
        Shape input tensors.
    activation : func
        Non-linear activation function.
    strides : list
        Convolutional layers strides.
    filters : list
        Convolutional layers number of filters.
    kernel_sizes : list
        Convolutional layers kernel sizes.

    Attributes
    ----------
    feature_extractor : nn.Module
        Neural network feature extractor block.
    """
    def __init__(self,
                 input_shape,
                 activation=nn.ReLU,
                 strides=[4, 2, 1],
                 filters=[32, 64, 64],
                 kernel_sizes=[8, 4, 3]):
        super(CNN, self).__init__()

        assert len(filters) == len(strides) and len(strides) == len(kernel_sizes)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        # Define feature extractor
        layers = []
        filters = [input_shape[0]] + filters
        for j in range(len(filters) - 1):
            layers += [init_(nn.Conv2d(
                filters[j], filters[j + 1], stride=strides[j],
                kernel_size=kernel_sizes[j])), activation()]
        self.feature_extractor = nn.Sequential(*layers)
        self.train()

    def forward(self, inputs):
        """
        Forward pass Neural Network

        Parameters
        ----------
        inputs : torch.tensor
            Input data.

        Returns
        -------
        out : torch.tensor
            Output feature map.
        """
        out = self.feature_extractor(inputs / 255.0)
        return out

# ---- Fixup Convolutional Neural Network -------------------------------------

class FixupCNN(nn.Module):
    """
    A larger version of the IMPALA CNN with Fixup init.
    See Fixup: https://arxiv.org/abs/1901.09321.

    Parameters
    ----------
    input_shape : tuple
        Shape input tensors.

    Attributes
    ----------
    feature_extractor : nn.Module
        Neural network feature extractor block.
    """
    def __init__(self, input_shape):
        super(FixupCNN, self).__init__()

        depth_in = input_shape[0]
        layers = []
        for depth_out in [16, 32, 32]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                FixupResidualModule(depth_out, 6),
                FixupResidualModule(depth_out, 6),
            ])
            depth_in = depth_out
        self.feature_extractor = nn.Sequential(*layers)
        self.train()

    def forward(self, inputs):
        """
        Forward pass Neural Network

        Parameters
        ----------
        inputs : torch.tensor
            Input data.

        Returns
        -------
        out : torch.tensor
            Output feature map.
        """
        out = self.feature_extractor(inputs / 255.0)
        return out

class FixupResidualModule(nn.Module):
    """
    FixupCNN resudial block.

    Parameters
    ----------
    depth : int
        Number of input feature maps.
    num_residual : int
        Number of residual feature blocks in the architecture
    """
    def __init__(self, depth, num_residual):
        super(FixupResidualModule, self).__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)

        for p in self.conv1.parameters():
            p.data.mul_(1 / math.sqrt(num_residual))
        for p in self.conv2.parameters():
            p.data.zero_()

        self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

    def forward(self, x):
        """
        Forward pass residual block

        Parameters
        ----------
        inputs : torch.tensor
            Input feature map.

        Returns
        -------
        out : torch.tensor
            Output feature map.
        """
        x = F.relu(x)
        out = x + self.bias1
        out = self.conv1(out)
        out = out + self.bias2
        out = F.relu(out)
        out = out + self.bias3
        out = self.conv2(out)
        out = out * self.scale
        out = out + self.bias4
        return out + x

# ---- get_model function -----------------------------------------------------

def get_model(name):
    """Returns model class from name."""
    if name == "MLP":
        return MLP
    elif name == "CNN":
        return CNN
    elif name == "Fixup":
        return FixupCNN
    else:
        raise ValueError("Specified model not found!")

