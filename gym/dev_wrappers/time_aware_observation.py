"""Wrapper for adding time aware observations to environment observation."""
from collections import OrderedDict

import jumpy as jp

import gym
import gym.spaces as spaces
from gym.core import ActType, ObsType
from gym.spaces import Box, Dict
from gym.vector import AsyncVectorEnv, SyncVectorEnv


class TimeAwareObservationV0(gym.ObservationWrapper):
    """Augment the observation with time information of the episode.

    Time can be represented as a normalized value between [0, 1]
    or by the number of timesteps remaining before truncation occurs.

    Example:
        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TimeAwareObservationV0(env)
        >>> env.observation_space
        Dict(obs: Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32), time: Box(0.0, 500, (1,), float32))
        >>> _ = env.reset()
        >>> env.step(env.action_space.sample())[0]
        OrderedDict([('obs',
        ...       array([ 0.02866629,  0.2310988 , -0.02614601, -0.2600732 ], dtype=float32)),
        ...      ('time', array([0.002]))])

    Flatten observation space example:
        >>> env = gym.make('CartPole-v1')
        >>> env = TimeAwareObservationV0(env, flatten=True)
        >>> env.observation_space
        Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38  0.0000000e+00], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38 500], (5,), float32)
        >>> _ = env.reset()
        >>> env.step(env.action_space.sample())[0]
        array([-0.01232257,  0.19335455, -0.02244143, -0.32388705,  0.002 ], dtype=float32)
    """

    def __init__(self, env: gym.Env, flatten=False, normalize_time=True):
        """Initialize :class:`TimeAwareObservationV0`.

        Args:
            env: The environment to apply the wrapper
            flatten: Flatten the observation to a `Box` of a single dimension
            normalize_time: if `True` return time in the range [0,1]
                otherwise return time as remaining timesteps before truncation
        """
        super().__init__(env)
        self.flatten = flatten
        self.normalize_time = normalize_time
        self.max_timesteps = self._get_max_timesteps(env)
        self.num_envs = getattr(env, "num_envs", 1)

        self.time_aware_observation_space = Dict(
            obs=env.observation_space, time=Box(0, self.max_timesteps, (self.num_envs,))
        )

        if self.flatten:
            self.observation_space = spaces.flatten_space(
                self.time_aware_observation_space
            )
        else:
            self.observation_space = self.time_aware_observation_space

    def observation(self, observation: ObsType):
        """Adds to the observation with the current time information.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time information appended to
        """
        time_observation = self._get_time_observation()
        observation = OrderedDict(obs=observation, time=time_observation)

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
        self.timesteps += 1
        observation, reward, terminated, truncated, info = super().step(action)

        self.timesteps = jp.where(terminated, 0, self.timesteps)

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment setting the time to zero.

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment
        """
        self.timesteps = jp.zeros(self.num_envs)
        return super().reset(**kwargs)

    def _get_time_observation(self):
        """Returns the `time` observation.

        If normalization is applied, a value in the
        range [0,1] is returned, otherwise, the
        remaining timesteps before truncation.
        """
        return (
            self.timesteps / self.max_timesteps
            if self.normalize_time
            else self.max_timesteps - self.timesteps
        )

    @staticmethod
    def _get_max_timesteps(env: gym.Env):
        """Get the max episode steps before truncation occurs.

        Args:
            env: the provided environment
        """
        return (
            env.get_attr("_max_episode_steps")[0]
            if isinstance(env, (AsyncVectorEnv, SyncVectorEnv))
            else getattr(env, "_max_episode_steps")
        )
