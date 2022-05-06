import gym


class OrderEnforcing(gym.Wrapper):
    """This will produce an error if `step` is called before an initial `reset`"""

    def __init__(self, env):
        super().__init__(env)
        self._has_reset = False

    def step(self, action):
        assert self._has_reset, "Cannot call env.step() before calling env.reset()"
        return self.env.step(action)

    def reset(self, **kwargs):
        self._has_reset = True
        return self.env.reset(**kwargs)

    def render(self, **kwargs):
        if hasattr(self.unwrapped, "disable_render_order_enforcing"):
            if not self.unwrapped.disable_render_order_enforcing:
                assert (
                    self._has_reset
                ), "Cannot call env.render() before calling env.reset()"
        else:
            assert self._has_reset, (
                "Cannot call env.render() before calling env.reset(), if this is a intended property, "
                "set `disable_render_order_enforcing=True` on the base environment (env.unwrapped)."
            )
        return self.env.render(**kwargs)
