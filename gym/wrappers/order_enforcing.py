"""Wrapper to enforce the order of environments."""
import gym


class OrderEnforcing(gym.Wrapper):
    """A wrapper that will produce an error if :meth:`step` is called before an initial :meth:`reset`.

    Example:
        >>> from gym.envs.classic_control import CartPoleEnv
        >>> env = CartPoleEnv()
        >>> env = OrderEnforcing(env)
        >>> env.step(0)
        ResetNeeded: Cannot call env.step() before calling env.reset()
        >>> env.render()
        ResetNeeded: Cannot call env.render() before calling env.reset()
        >>> env.reset()
        >>> env.render()
        >>> env.step(0)
    """

    def __init__(self, env):
        """A wrapper that will produce an error if :meth:`step` is called before an initial :meth:`reset`."""
        super().__init__(env)
        self._has_reset = False

    def step(self, action):
        """Steps through the environment with :param:`kwargs`."""
        assert self._has_reset, "Cannot call env.step() before calling env.reset()"
        return self.env.step(action)

    def reset(self, **kwargs):
        """Resets the environment with :param:`kwargs`."""
        self._has_reset = True
        return self.env.reset(**kwargs)

    def render(self, **kwargs):
        """Checks that the environment has been :meth:`reset` before rendering the environment."""
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
