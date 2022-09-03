"""A passive environment checker wrapper for an environment's observation and action space along with the reset, step and render functions."""
import gym
from gym.core import ActType
from gym.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)


class PassiveEnvChecker(gym.Wrapper):
    """A passive environment checker wrapper that surrounds the step, reset and render functions to check they follow the gym API."""

    def __init__(self, env):
        """Initialises the wrapper with the environments, run the observation and action space tests."""
        super().__init__(env)

        assert hasattr(
            env, "action_space"
        ), "The environment must specify an action space. https://www.gymlibrary.dev/content/environment_creation/"
        check_action_space(env.action_space)
        assert hasattr(
            env, "observation_space"
        ), "The environment must specify an observation space. https://www.gymlibrary.dev/content/environment_creation/"
        check_observation_space(env.observation_space)

        self.checked_reset = False
        self.checked_step = False
        self.checked_render = False

    def step(self, action: ActType):
        """Steps through the environment that on the first call will run the `passive_env_step_check`."""
        if self.checked_step is False:
            self.checked_step = True
            return env_step_passive_checker(self.env, action)
        else:
            return self.env.step(action)

    def reset(self, **kwargs):
        """Resets the environment that on the first call will run the `passive_env_reset_check`."""
        if self.checked_reset is False:
            self.checked_reset = True
            return env_reset_passive_checker(self.env, **kwargs)
        else:
            return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        """Renders the environment that on the first call will run the `passive_env_render_check`."""
        if self.checked_render is False:
            self.checked_render = True
            return env_render_passive_checker(self.env, *args, **kwargs)
        else:
            return self.env.render(*args, **kwargs)
