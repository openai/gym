import numpy as np

from gym import ObservationWrapper
from gym.spaces import Box


class GrayScaleObservation(ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale."""

    def __init__(self, env, keep_dim=False):
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
        grayscale_obs = (
            observation[..., 0] * 0.2126
            + observation[..., 1] * 0.587
            + observation[..., 2] * 0.114
        )
        if self.keep_dim:
            grayscale_obs = np.expand_dims(grayscale_obs, -1)
        return grayscale_obs
