import gym


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None, done_on_limit=True):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._done_on_limit = done_on_limit
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            if self._done_on_limit:
                done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
