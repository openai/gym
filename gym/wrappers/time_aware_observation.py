import numpy as np

from gym import ObservationWrapper
from gym.spaces import Box
from gym.utils.step_api_compatibility import step_api_compatibility


class TimeAwareObservation(ObservationWrapper):
    r"""Augment the observation with current time step in the trajectory.

    .. note::
        Currently it only works with one-dimensional observation space. It doesn't
        support pixel observation space yet.

    """

    def __init__(self, env, new_step_api=False):
        super().__init__(env, new_step_api)
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.dtype == np.float32
        low = np.append(self.observation_space.low, 0.0)
        high = np.append(self.observation_space.high, np.inf)
        self.observation_space = Box(low, high, dtype=np.float32)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def observation(self, observation):
        return np.append(observation, self.t)

    def step(self, action):
        self.t += 1
        return step_api_compatibility(
            super().step(action), self.new_step_api, self.is_vector_env
        )

    def reset(self, **kwargs):
        self.t = 0
        return super().reset(**kwargs)
