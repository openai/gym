from typing import Optional

import gym


class TimeLimit(gym.Wrapper):
    """This wrapper will issue a `done` signal if a maximum number of timesteps is exceeded.

    Oftentimes, it is **very** important to distinguish `done` signals that were produced by the
    `TimeLimit` wrapper (truncations) and those that originate from the underlying environment (terminations).
    This can be done by looking at the `info` that is returned when `done` signal was issued.

    The done-signal originates from the time limit (i.e. it signifies a *truncation*) if and only if
    the key `"TimeLimit.truncated"` exists in `info` and the corresponding value is `True`.

    Args:
        env: The environment that will be wrapped
        max_episode_steps (Optional[int]): The maximum number of steps until a done-signal occurs. If it is `None`, the value from `env.spec` (if available) will be used
    """

    def __init__(self, env, max_episode_steps: Optional[int] = None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
