import numpy as np

from gym.spaces import Box
from gym import ObservationWrapper


class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image. """
    def __init__(self, env, size):
        super(ResizeObservation, self).__init__(env)
        assert size > 0
        self.size = size

        shape = (self.size, self.size) + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, observation):
        import cv2
        observation = cv2.resize(observation, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation
