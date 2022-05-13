"""A wrapper for filtering dictionary observations by their keys."""
import copy
from typing import Sequence

import gym
from gym import spaces


class FilterObservation(gym.ObservationWrapper):
    """Filter Dict observation space by the keys.

    Example:
        >>> import gym
        >>> env = gym.wrappers.TransformObservation(
        ...     gym.make('CartPole-v1'), lambda obs: {'obs': obs, 'time': 0}
        ... )
        >>> env.observation_space = gym.spaces.Dict(obs=env.observation_space, time=gym.spaces.Discrete(1))
        >>> env.reset()
        {'obs': array([-0.00067088, -0.01860439,  0.04772898, -0.01911527], dtype=float32), 'time': 0}
        >>> env = FilterObservation(env, filter_keys=['time'])
        >>> env.reset()
        {'obs': array([ 0.04560107,  0.04466959, -0.0328232 , -0.02367178], dtype=float32)}
        >>> env.step(0)
        ({'obs': array([ 0.04649447, -0.14996664, -0.03329664,  0.25847703], dtype=float32)}, 1.0, False, {})
    """

    def __init__(self, env: gym.Env, filter_keys: Sequence[str] = None):
        """A wrapper that filters dictionary observations by their keys.

        Args:
            env: The environment to apply the wrapper
            filter_keys: List of keys to be included in the observations. If ``None``, observations will not be filtered and this wrapper has no effect

        Raises:
            ValueError: If the environment's observation space is not :class:`spaces.Dict`
            ValueError: If any of the `filter_keys` are not included in the original `env`'s observation space
        """
        super().__init__(env)

        wrapped_observation_space = env.observation_space
        if not isinstance(wrapped_observation_space, spaces.Dict):
            raise ValueError(
                f"FilterObservationWrapper is only usable with dict observations, "
                f"environment observation space is {type(wrapped_observation_space)}"
            )

        observation_keys = wrapped_observation_space.spaces.keys()
        if filter_keys is None:
            filter_keys = tuple(observation_keys)

        missing_keys = {key for key in filter_keys if key not in observation_keys}
        if missing_keys:
            raise ValueError(
                "All the filter_keys must be included in the original observation space.\n"
                f"Filter keys: {filter_keys}\n"
                f"Observation keys: {observation_keys}\n"
                f"Missing keys: {missing_keys}"
            )

        self.observation_space = type(wrapped_observation_space)(
            [
                (name, copy.deepcopy(space))
                for name, space in wrapped_observation_space.spaces.items()
                if name in filter_keys
            ]
        )

        self._env = env
        self._filter_keys = tuple(filter_keys)

    def observation(self, observation):
        """Filters the observations.

        Args:
            observation: The observation to filter

        Returns:
            The filtered observations
        """
        filter_observation = self._filter_observation(observation)
        return filter_observation

    def _filter_observation(self, observation):
        observation = type(observation)(
            [
                (name, value)
                for name, value in observation.items()
                if name in self._filter_keys
            ]
        )
        return observation
