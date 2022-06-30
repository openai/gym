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
        """A wrapper that will produce an error if :meth:`step` is called before an initial :meth:`reset`.

        Args:
            env: The environment to wrap
            disable_render_order_enforcing: If to disable render order enforcing
        """
        super().__init__(env)
        self._has_reset: bool = False
        self._disable_render_order_enforcing: bool = disable_render_order_enforcing

    def step(self, action):
        """Steps through the environment with `kwargs`."""
        if not self._has_reset:
            raise ResetNeeded("Cannot call env.step() before calling env.reset()")
        return self.env.step(action)

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""
        self._has_reset = True
        return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        """Renders the environment with `kwargs`."""
        if not self._disable_render_order_enforcing and not self._has_reset:
            raise ResetNeeded(
                "Cannot call `env.render()` before calling `env.reset()`, if this is a intended action, "
                "set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper."
            )
        return self.env.render(*args, **kwargs)

    @property
    def has_reset(self):
        """Returns if the environment has been reset before."""
        return self._has_reset
