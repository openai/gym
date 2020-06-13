import numpy as np

import gym
from gym import spaces


def rescale_values(values, old_low, old_high, new_low, new_high):
    rescaled_values = new_low + (new_high - new_low) * (
        (values - old_low) / (old_high - old_low))
    rescaled_values = np.clip(rescaled_values, new_low, new_high)
    return rescaled_values


def verify_observation_space_type(observation_space):
    if not isinstance(observation_space, spaces.Box):
        raise TypeError("Expected Box observation space. Got: {}"
                        "".format(type(observation_space)))


def verify_observation_space_bounds(observation_space):
    if np.any(~np.isfinite((
            observation_space.low, observation_space.high))):
        raise ValueError(
            "Observation space 'low' and 'high' need to be finite."
            " Got: observation_space.low={}, observation_space.high={}"
            "".format(observation_space.low, observation_space.high))


def rescale_box_space(observation_space, low, high):
    shape = observation_space.shape
    dtype = observation_space.dtype

    new_low = low + np.zeros(shape, dtype=dtype)
    new_high = high + np.zeros(shape, dtype=dtype)

    observation_space = spaces.Box(
        low=new_low, high=new_high, shape=shape, dtype=dtype)

    return observation_space


class RescaleObservation(gym.ObservationWrapper):
    def __init__(self, env, low, high):
        r"""Rescale observation space to a range [`low`, `high`].
        Example:
            >>> RescaleObservation(env, low, high).observation_space == Box(low, high)
            True
        Raises:
            TypeError: If `not isinstance(environment.observation_space, (Box, Tuple, Dict))`.
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

        if isinstance(env.observation_space, spaces.Box):
            verify_observation_space_type(env.observation_space)
            verify_observation_space_bounds(env.observation_space)
            self.observation_space = rescale_box_space(
                env.observation_space, low, high)
        elif isinstance(env.observation_space, spaces.Tuple):
            for observation_space in env.observation_space.spaces:
                verify_observation_space_type(observation_space)
                verify_observation_space_bounds(observation_space)
            self.observation_space = spaces.Tuple([
                rescale_box_space(observation_space, low, high)
                for observation_space
                in env.observation_space.spaces
            ])
        elif isinstance(env.observation_space, spaces.Dict):
            for observation_space in env.observation_space.spaces.values():
                verify_observation_space_type(observation_space)
                verify_observation_space_bounds(observation_space)
            self.observation_space = spaces.Dict({
                name: rescale_box_space(observation_space, low, high)
                for name, observation_space
                in env.observation_space.spaces.items()
            })
        else:
            raise TypeError("Unsupported observation space type: {}"
                            "".format(type(env.observation_space)))

    def observation(self, observation):
        if isinstance(self.observation_space, spaces.Box):
            rescaled_observation = rescale_values(
                observation,
                old_low=self.env.observation_space.low,
                old_high=self.env.observation_space.high,
                new_low=self.observation_space.low,
                new_high=self.observation_space.high)
        elif isinstance(self.observation_space, spaces.Tuple):
            rescaled_observation = type(observation)((
                rescale_values(
                    value,
                    old_low=self.env.observation_space[i].low,
                    old_high=self.env.observation_space[i].high,
                    new_low=self.observation_space[i].low,
                    new_high=self.observation_space[i].high)
                for i, value in enumerate(observation)
            ))
        elif isinstance(self.observation_space, spaces.Dict):
            rescaled_observation = type(observation)((
                (key, rescale_values(
                    value,
                    old_low=self.env.observation_space[key].low,
                    old_high=self.env.observation_space[key].high,
                    new_low=self.observation_space[key].low,
                    new_high=self.observation_space[key].high))
                for key, value in observation.items()
            ))
        else:
            raise TypeError("Unsupported observation space type: {}"
                            "".format(type(self.env.observation_space)))

        return rescaled_observation
