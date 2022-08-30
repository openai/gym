"""Wrapper for limiting the time steps of an environment."""
from typing import Optional

import gym


class TimeLimit(gym.Wrapper):
    """This wrapper will issue a `truncated` signal if a maximum number of timesteps is exceeded.

    If a truncation is not defined inside the environment itself, this is the only place that the truncation signal is issued.
    Critically, this is different from the `terminated` signal that originates from the underlying environment as part of the MDP.

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
    ):
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
            The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
