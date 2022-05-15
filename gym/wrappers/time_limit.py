"""Wrapper for limiting the time steps of an environment."""
from typing import Optional

import gym


class TimeLimit(gym.Wrapper):
    """This wrapper will issue a `done` signal if a maximum number of timesteps is exceeded.

    Oftentimes, it is **very** important to distinguish `done` signals that were produced by the
    :class:`TimeLimit` wrapper (truncations) and those that originate from the underlying environment (terminations).
    This can be done by looking at the ``info`` that is returned when `done`-signal was issued.
    The done-signal originates from the time limit (i.e. it signifies a *truncation*) if and only if
    the key `"TimeLimit.truncated"` exists in ``info`` and the corresponding value is ``True``.

    Example:
       >>> from gym.envs.classic_control import CartPoleEnv
       >>> from gym.wrappers import TimeLimit
       >>> env = CartPoleEnv()
       >>> env = TimeLimit(env, max_episode_steps=1000)
    """

    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: An optional max episode steps (if ``Ǹone``, ``env.spec.max_episode_steps`` is used)
        """
        super().__init__(env)
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
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
