import numpy as np

import gym
from gym import spaces


def rescale_values(values, old_low, old_high, new_low, new_high):
    rescaled_values = new_low + (new_high - new_low) * (
        (values - old_low) / (old_high - old_low))
    rescaled_values = np.clip(rescaled_values, new_low, new_high)
    return rescaled_values


class RescaleObservation(gym.ObservationWrapper):
    def __init__(self, env, low, high):
        r"""Rescale observation space to a range [`low`, `high`].

        Example:
            >>> RescaleObservation(env, low, high).observation_space == Box(low, high)
            True

        Raises:
            TypeError: If `not isinstance(environment.observation_space, spaces.Box)`.
            ValueError: If either `low` or `high` is not finite.
            ValueError: If any of `observation_space.{low,high}` is not finite.
            ValueError: If `high <= low`.
        """
        if np.any(~np.isfinite((low, high))):
            raise ValueError(
                "Arguments 'low' and 'high' need to be finite."
                " Got: low={}, high={}".format(low, high))

        if np.any(high <= low):
            raise ValueError("Argument `low` must be smaller than `high`"
                             " Got: low={}, high=".format(low, high))

        super(RescaleObservation, self).__init__(env)

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("Expected Box observation space. Got: {}"
                            "".format(type(env.observation_space)))

        if np.any(~np.isfinite((
                env.observation_space.low, env.observation_space.high))):
            raise ValueError(
                "Observation space 'low' and 'high' need to be finite."
                " Got: observation_space.low={}, observation_space.high={}"
                "".format(env.observation_space.low,
                          env.observation_space.high))

        shape = env.observation_space.shape
        dtype = env.observation_space.dtype

        self.low = low + np.zeros(shape, dtype=dtype)
        self.high = high + np.zeros(shape, dtype=dtype)
        self.observation_space = spaces.Box(
            low=self.low, high=self.high, shape=shape, dtype=dtype)

    def observation(self, observation):
        rescaled_observation = rescale_values(
            observation,
            old_low=self.env.observation_space.low,
            old_high=self.env.observation_space.high,
            new_low=self.low,
            new_high=self.high)

        return rescaled_observation
