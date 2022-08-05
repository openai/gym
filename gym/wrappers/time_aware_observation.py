"""Wrapper for adding time aware observations to environment observation."""
import numpy as np

import gym
from gym.spaces import Box


class TimeAwareObservation(gym.ObservationWrapper):
    """Augment the observation with the current time step in the episode.

    The observation space of the wrapped environment is assumed to be a flat :class:`Box`.
    In particular, pixel observations are not supported. This wrapper will append the current timestep within the current episode to the observation.

    Example:
        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TimeAwareObservation(env)
        >>> env.reset()
        array([ 0.03810719,  0.03522411,  0.02231044, -0.01088205,  0.        ])
        >>> env.step(env.action_space.sample())[0]
        array([ 0.03881167, -0.16021058,  0.0220928 ,  0.28875574,  1.        ])
    """

    def __init__(self, env: gym.Env):
        """Initialize :class:`TimeAwareObservation` that requires an environment with a flat :class:`Box` observation space.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.dtype == np.float32
        low = np.append(self.observation_space.low, 0.0)
        high = np.append(self.observation_space.high, np.inf)
        self.observation_space = Box(low, high, dtype=np.float32)
        self.is_vector_env = getattr(env, "is_vector_env", False)

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
