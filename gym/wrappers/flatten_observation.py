from gym.spaces import flatten
from gym import ObservationWrapper


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def observation(self, observation):
        return flatten(self.observation_space, observation)
