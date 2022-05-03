import gym


class OrderEnforcing(gym.Wrapper):
    """This will produce an error if `step` is called before an initial `reset`"""

    def __init__(self, env):
        super().__init__(env)
        self._has_reset = False

    def step(self, action):
        assert self._has_reset, "Cannot call env.step() before calling reset()"
        return self.env.step(action)

    def reset(self, **kwargs):
        self._has_reset = True
        return self.env.reset(**kwargs)
