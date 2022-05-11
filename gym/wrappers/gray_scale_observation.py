"""Wrapper that converts a colour observation to greyscale."""
import numpy as np

import gym
from gym import ObservationWrapper
from gym.spaces import Box


class GrayScaleObservation(ObservationWrapper):
    """Convert the image observation from RGB to gray scale.

    Example:
        >>> env = gym.make('CarRacing-v1')
        >>> env.reset()
        (96, 96, 3)
        >>> env = gym.wrappers.GrayScaleObservation(gym.make('CarRacing-v1'))
        (96, 96)
        >>> env = gym.wrappers.GrayScaleObservation(gym.make('CarRacing-v1'), keep_dim=True)
        (96, 96, 1)
    """

    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. observations are of the shape AxBx1.
                Otherwise, they are of shape AxB.
        """
        super().__init__(env)
        self.keep_dim = keep_dim

        assert (
            len(env.observation_space.shape) == 3
            and env.observation_space.shape[-1] == 3
        )

        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = Box(
                low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
            )
        else:
            self.observation_space = Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

    def observation(self, observation):
        """Converts the colour observation to greyscale.

        Args:
            observation: Colour observations

        Returns:
            Grayscale observations
        """
        import cv2

        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation
