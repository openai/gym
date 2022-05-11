"""Wrapper for adding time aware observations to environment observation."""
import numpy as np

import gym
from gym import ObservationWrapper
from gym.spaces import Box


class TimeAwareObservation(ObservationWrapper):
    """Augment the observation with current time step in the trajectory.

    Note:
        Currently it only works with one-dimensional observation space.
        It doesn't support pixel observation space yet.

    Example:
        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TimeAwareObservation(env)
        >>> env.reset()
        array([ 0.01746378, -0.0495109 , -0.01070071, -0.03747902,  0.        ])
    """

    def __init__(self, env: gym.Env):
        """Initialise time aware observation that requires an environment with a Box observation space.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.dtype == np.float32
        low = np.append(self.observation_space.low, 0.0)
        high = np.append(self.observation_space.high, np.inf)
        self.observation_space = Box(low, high, dtype=np.float32)

    def observation(self, observation):
        """Adds to the observation with the current time step.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time step appended to
        """
        return np.append(observation, self.t)

    def step(self, action):
        """Steps through the environment, incrementing the time step.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        self.t += 1
        return super().step(action)

    def reset(self, **kwargs):
        """Reset the environment setting the time to zero.

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment
        """
        self.t = 0
        return super().reset(**kwargs)
