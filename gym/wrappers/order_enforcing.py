"""Wrapper to enforce the proper ordering of environment operations."""
import gym
from gym.error import ResetNeeded


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

    def __init__(self, env: gym.Env, disable_render_order_enforcing: bool = False):
        """A wrapper that will produce an error if :meth:`step` is called before an initial :meth:`reset`."""
        super().__init__(env)
        self._has_reset = False
        self._disable_render_order_enforcing = disable_render_order_enforcing

    def step(self, action):
        """Steps through the environment with `kwargs`."""
        if not self._has_reset:
            raise ResetNeeded("Cannot call env.step() before calling env.reset()")
        return self.env.step(action)

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""
        self._has_reset = True
        return self.env.reset(**kwargs)

    def render(self, **kwargs):
        if self._disable_render_order_enforcing is False and not self._has_reset:
            raise ResetNeeded(
                "Cannot call env.render() before calling env.reset(), if this is a intended action, "
                "set `disable_render_order_enforcing` on the OrderEnforcer wrapper = False."
            )
        return self.env.render(**kwargs)
