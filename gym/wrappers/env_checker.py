"""A passive environment checker wrapper for an environment's observation and action space along with the reset, step and render functions."""
from typing import Tuple, Union

import gym
from gym.core import ActType, ObsType
from gym.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    passive_env_render_check,
    passive_env_reset_check,
    passive_env_step_check,
)


class PassiveEnvChecker(gym.Wrapper):
    """A passive environment checker wrapper that surrounds the step, reset and render functions to check they follow the gym API."""

    def __init__(self, env):
        """Initialises the wrapper with the environments, run the observation and action space tests."""
        super().__init__(env)

        assert hasattr(
            env, "action_space"
        ), "You must specify a action space. https://www.gymlibrary.ml/content/environment_creation/"
        check_observation_space(env.action_space)
        assert hasattr(
            env, "observation_space"
        ), "You must specify an observation space. https://www.gymlibrary.ml/content/environment_creation/"
        check_action_space(env.observation_space)

        self.checked_reset = False
        self.checked_step = False
        self.checked_render = False

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """Steps through the environment that on the first call will run the `passive_env_step_check`."""
        if self.checked_step is False:
            self.checked_step = True
            return passive_env_step_check(self.env, action)
        else:
            return self.env.step(action)

    def reset(self, **kwargs) -> Union[ObsType, Tuple[ObsType, dict]]:
        """Resets the environment that on the first call will run the `passive_env_reset_check`."""
        if self.checked_reset is False:
            self.checked_reset = True
            return passive_env_reset_check(self.env, **kwargs)
        else:
            return self.env.reset(**kwargs)

    def render(self, **kwargs):
        """Renders the environment that on the first call will run the `passive_env_render_check`."""
        if self.checked_render is False:
            self.checked_render = True
            return passive_env_render_check(self.env, **kwargs)
        else:
            return self.env.render(**kwargs)
