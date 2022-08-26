"""Wrapper for limiting the time steps of an environment."""
from typing import Optional

import gym
from gym.utils.step_api_compatibility import step_api_compatibility


class TimeLimit(gym.Wrapper):
    """This wrapper will issue a `truncated` signal if a maximum number of timesteps is exceeded.

    If a truncation is not defined inside the environment itself, this is the only place that the truncation signal is issued.
    Critically, this is different from the `terminated` signal that originates from the underlying environment as part of the MDP.

    (deprecated)
    This information is passed through ``info`` that is returned when `done`-signal was issued.
    The done-signal originates from the time limit (i.e. it signifies a *truncation*) if and only if
    the key `"TimeLimit.truncated"` exists in ``info`` and the corresponding value is ``True``. This will be removed in favour
    of only issuing a `truncated` signal in future versions.

    Example:
       >>> from gym.envs.classic_control import CartPoleEnv
       >>> from gym.wrappers import TimeLimit
       >>> env = CartPoleEnv()
       >>> env = TimeLimit(env, max_episode_steps=1000)
    """

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: Optional[int] = None,
        new_step_api: bool = False,
    ):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: An optional max episode steps (if ``None``, ``env.spec.max_episode_steps`` is used)
            new_step_api (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
        """
        super().__init__(env, new_step_api)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, done, info)`` with "TimeLimit.truncated"=True
            when truncated (the number of steps elapsed >= max episode steps) or
            "TimeLimit.truncated"=False if the environment terminated
        """
        observation, reward, terminated, truncated, info = step_api_compatibility(
            self.env.step(action),
            True,
        )
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            if self.new_step_api is True or terminated is False:
                # As the old step api cannot encode both terminated and truncated, we favor terminated in the case of both.
                #   Therefore, if new step api (i.e. not old step api) or when terminated is False to prevent the overriding
                truncated = True

        return step_api_compatibility(
            (observation, reward, terminated, truncated, info),
            self.new_step_api,
        )

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
