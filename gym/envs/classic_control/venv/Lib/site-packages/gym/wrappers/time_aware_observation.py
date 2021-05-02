import numpy as np

from gym.spaces import Box
from gym import ObservationWrapper


class TimeAwareObservation(ObservationWrapper):
    r"""Augment the observation with current time step in the trajectory. 

    .. note::
        Currently it only works with one-dimensional observation space. It doesn't
        support pixel observation space yet. 

    """
    def __init__(self, env):
        super(TimeAwareObservation, self).__init__(env)
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.dtype == np.float32
        low = np.append(self.observation_space.low, 0.0)
        high = np.append(self.observation_space.high, np.inf)
        self.observation_space = Box(low, high, dtype=np.float32)

    def observation(self, observation):
        return np.append(observation, self.t)

    def step(self, action):
        self.t += 1
        return super(TimeAwareObservation, self).step(action)

    def reset(self, **kwargs):
        self.t = 0
        return super(TimeAwareObservation, self).reset(**kwargs)
