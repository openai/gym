import numpy as np

from gym import ObservationWrapper
from gym.spaces import Box

try:
    import tinyscaler
except ImportError:
    tinyscaler = None


class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image."""

    def __init__(self, env, shape):
        if tinyscaler is None:
            raise ImportError(
                "Tinyscaler is not installed, Try run `pip install gym[other]` to get dependencies"
            )

        super().__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape

        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = tinyscaler.scale(observation, self.shape[::-1])
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation
