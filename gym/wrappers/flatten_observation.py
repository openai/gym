import gym.spaces as spaces
from gym import ObservationWrapper
import warnings


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)
        warnings.warn(
            "Gym's internal preprocessing wrappers are now deprecated. While they will continue to work for the foreseeable future, we strongly recommend using SuperSuit instead: https://github.com/PettingZoo-Team/SuperSuit"
        )

    def observation(self, observation):
        return spaces.flatten(self.env.observation_space, observation)
