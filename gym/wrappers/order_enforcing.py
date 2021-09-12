import gym


class OrderEnforcing(gym.Wrapper):
    def __init__(self, env):
        super(OrderEnforcing, self).__init__(env)
        self._has_reset = False

    def step(self, action):
        assert self._has_reset, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._has_reset = True
        return self.env.reset(**kwargs)
