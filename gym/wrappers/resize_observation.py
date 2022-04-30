from typing import Union

import numpy as np

from gym import ObservationWrapper
from gym.spaces import Box


class ResizeObservation(ObservationWrapper):
    """Resize the image observation.

    This wrapper works on environments with image observations (or more generally observations of shape AxBxC) and resizes
    the observation to the shape given by the 2-tuple `shape`. The argument `shape` may also be an integer. In that case, the
    observation is scaled to a square of side-length `shape`.

    Args:
        shape (Union[tuple, int]): The dimensions of the resized observation
    """

    def __init__(self, env, shape: Union[tuple, int]):
        super().__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape

        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        import cv2

        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation
