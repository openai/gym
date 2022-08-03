"""Wrapper for adding time aware observations to environment observation."""
from collections import OrderedDict

import jumpy as jp

import gym
import gym.spaces as spaces
from gym.core import ActType, ObsType
from gym.spaces import Box, Dict


class TimeAwareObservationV0(gym.ObservationWrapper):
    """Augment the observation with the current time step in the episode.

    Example:
        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TimeAwareObservationV0(env)
        >>> env.reset()
        Dict(obs: Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32), time: Box(0.0, inf, (1,), float32))
        >>> env.step(env.action_space.sample())[0]
        OrderedDict([('time', array([1.])),
        ...  ('obs',
        ...    array([ 0.02768888,  0.1745313 ,  0.03663293, -0.32239535], dtype=float32))])
    """

    def __init__(self, env: gym.Env, flatten=False):
        """Initialize :class:`TimeAwareObservationV0`.

        Args:
            env: The environment to apply the wrapper
            flatten: Flatten the observation to a `Box` of a single dimension
        """
        super().__init__(env)
        self.flatten = flatten
        self.num_envs = getattr(env, "num_envs", 1)

        self.time_aware_observation_space = Dict(
            obs=env.observation_space, time=Box(0, jp.inf, (self.num_envs,))
        )

        if self.flatten:
            self.observation_space = spaces.flatten_space(
                self.time_aware_observation_space
            )
        else:
            self.observation_space = self.time_aware_observation_space

    def observation(self, observation: ObsType):
        """Adds to the observation with the current time step.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time step appended to
        """
        observation = OrderedDict(obs=observation, time=self.t)

        return (
            spaces.flatten(self.time_aware_observation_space, observation)
            if self.flatten
            else observation
        )

    def step(self, action: ActType):
        """Steps through the environment, incrementing the time step.

        In vectorized environments, if any of the environments
        reach a terminal state, set the respective time value to zero.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        self.t += 1
        observation, reward, terminated, truncated, info = super().step(action)

        self.t = jp.where(terminated, 0, self.t)

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment setting the time to zero.

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment
        """
        self.t = jp.zeros(self.num_envs)
        return super().reset(**kwargs)
