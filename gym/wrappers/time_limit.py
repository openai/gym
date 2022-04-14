from typing import Optional

import gym


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        step_returns = self.env.step(action)
        if len(step_returns) == 4:
            observation, reward, done, info = self.env.step(action)
            if self._elapsed_steps >= self._max_episode_steps:
                info["TimeLimit.truncated"] = not done
                done = True
            return observation, reward, done, info
        else:
            observation, reward, terminated, truncated, info = step_returns
            self._elapsed_steps += 1
            if self._elapsed_steps >= self._max_episode_steps:
                truncated = True
                info["TimeLimit.truncated"] = truncated
            return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
