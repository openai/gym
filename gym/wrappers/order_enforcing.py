import gym
from gym.error import ResetNeeded


class OrderEnforcing(gym.Wrapper):
    """This will produce an error if `step` is called before an initial `reset`"""

    def __init__(self, env: gym.Env, disable_render_order_enforcing: bool = False):
        super().__init__(env)
        self._has_reset = False
        self._disable_render_order_enforcing = disable_render_order_enforcing

    def step(self, action):
        if not self._has_reset:
            raise ResetNeeded("Cannot call env.step() before calling env.reset()")
        return self.env.step(action)

    def reset(self, **kwargs):
        self._has_reset = True
        return self.env.reset(**kwargs)

    def render(self, **kwargs):
        if self._disable_render_order_enforcing is False and not self._has_reset:
            raise ResetNeeded("Cannot call env.render() before calling env.reset(), if this is a intended action, "
                              "set `disable_render_order_enforcing` on the OrderEnforcer wrapper = False.")
        return self.env.render(**kwargs)
