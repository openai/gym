"""Wrapper for resizing observations."""
from typing import Union

import numpy as np

import gym
from gym.error import DependencyNotInstalled
from gym.spaces import Box


class ResizeObservation(gym.ObservationWrapper):
    """Resize the image observation.

    This wrapper works on environments with image observations (or more generally observations of shape AxBxC) and resizes
    the observation to the shape given by the 2-tuple :attr:`shape`. The argument :attr:`shape` may also be an integer.
    In that case, the observation is scaled to a square of side-length :attr:`shape`.

    Example:
        >>> import gym
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = ResizeObservation(env, 64)
        >>> env.observation_space.shape
        (64, 64, 3)
    """

    def __init__(self, env: gym.Env, shape: Union[tuple, int]):
        """Resizes image observations to shape given by :attr:`shape`.

        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        super().__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape

        self.shape = tuple(shape)

        assert isinstance(
            env.observation_space, Box
        ), f"Expected the observation space to be Box, actual type: {type(env.observation_space)}"
        obs_shape = self.shape + env.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """
        try:
            import cv2
        except ImportError:
            raise DependencyNotInstalled(
                "opencv is not install, run `pip install gym[other]`"
            )

        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation
