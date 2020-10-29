"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import enum

import numpy as np
import gym

import rrc_simulation
from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.tasks import move_cube
from rrc_simulation import visual_objects
from rrc_simulation import TriFingerPlatform


class FlatObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = [
            self.observation_space[name].low.flatten()
            for name in self.observation_names
        ]

        high = [
            self.observation_space[name].high.flatten()
            for name in self.observation_names
        ]

        self.observation_space = gym.spaces.Box(
            low=np.concatenate(low), high=np.concatenate(high)
        )

    def observation(self, obs):

        observation = [obs[name].flatten() for name in self.observation_names]

        observation = np.concatenate(observation)
        return observation