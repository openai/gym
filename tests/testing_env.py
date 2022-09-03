"""Provides a generic testing environment for use in tests with custom reset, step and render functions."""
import types
from typing import Any, Dict, Optional, Tuple, Union

import gym
from gym import spaces
from gym.core import ActType, ObsType
from gym.envs.registration import EnvSpec


def basic_reset_fn(
    self,
    *,
    seed: Optional[int] = None,
    options: Optional[dict] = None,
) -> Union[ObsType, Tuple[ObsType, dict]]:
    """A basic reset function that will pass the environment check using random actions from the observation space."""
    super(GenericTestEnv, self).reset(seed=seed)
    self.observation_space.seed(seed)
    return self.observation_space.sample(), {"options": options}


def new_step_fn(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
    """A step function that follows the new step api that will pass the environment check using random actions from the observation space."""
    return self.observation_space.sample(), 0, False, False, {}


def old_step_fn(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
    """A step function that follows the old step api that will pass the environment check using random actions from the observation space."""
    return self.observation_space.sample(), 0, False, {}


def basic_render_fn(self):
    """Basic render fn that does nothing."""
    pass


# todo: change all testing environment to this generic class
class GenericTestEnv(gym.Env):
    """A generic testing environment for use in testing with modified environments are required."""

    def __init__(
        self,
        action_space: spaces.Space = spaces.Box(0, 1, (1,)),
        observation_space: spaces.Space = spaces.Box(0, 1, (1,)),
        reset_fn: callable = basic_reset_fn,
        step_fn: callable = new_step_fn,
        render_fn: callable = basic_render_fn,
        metadata: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        spec: EnvSpec = EnvSpec("TestingEnv-v0", "testing-env-no-entry-point"),
    ):
        self.metadata = {} if metadata is None else metadata
        self.render_mode = render_mode
        self.spec = spec

        if observation_space is not None:
            self.observation_space = observation_space
        if action_space is not None:
            self.action_space = action_space

        if reset_fn is not None:
            self.reset = types.MethodType(reset_fn, self)
        if step_fn is not None:
            self.step = types.MethodType(step_fn, self)
        if render_fn is not None:
            self.render = types.MethodType(render_fn, self)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        # If you need a default working reset function, use `basic_reset_fn` above
        raise NotImplementedError("TestingEnv reset_fn is not set.")

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        raise NotImplementedError("TestingEnv step_fn is not set.")

    def render(self):
        raise NotImplementedError("testingEnv render_fn is not set.")
