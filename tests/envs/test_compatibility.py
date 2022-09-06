import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

import gym
from gym.spaces import Discrete
from gym.wrappers.compatibility import EnvCompatibility, LegacyEnv


class LegacyEnvExplicit(LegacyEnv, gym.Env):
    """Legacy env that explicitly implements the old API."""

    observation_space = Discrete(1)
    action_space = Discrete(1)
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        pass

    def reset(self):
        return 0

    def step(self, action):
        return 0, 0, False, {}

    def render(self, mode="human"):
        if mode == "human":
            return
        elif mode == "rgb_array":
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        pass

    def seed(self, seed=None):
        pass


class LegacyEnvImplicit(gym.Env):
    """Legacy env that implicitly implements the old API as a protocol."""

    observation_space = Discrete(1)
    action_space = Discrete(1)
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        pass

    def reset(self):  # type: ignore
        return 0  # type: ignore

    def step(self, action: Any) -> Tuple[int, float, bool, Dict]:
        return 0, 0.0, False, {}

    def render(self, mode: Optional[str] = "human") -> Any:
        if mode == "human":
            return
        elif mode == "rgb_array":
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        pass

    def seed(self, seed: Optional[int] = None):
        pass


def test_explicit():
    old_env = LegacyEnvExplicit()
    assert isinstance(old_env, LegacyEnv)
    env = EnvCompatibility(old_env, render_mode="rgb_array")
    assert env.observation_space == Discrete(1)
    assert env.action_space == Discrete(1)
    assert env.reset() == (0, {})
    assert env.reset(seed=0, options={"some": "option"}) == (0, {})
    assert env.step(0) == (0, 0, False, False, {})
    assert env.render().shape == (1, 1, 3)
    env.close()


def test_implicit():
    old_env = LegacyEnvImplicit()
    if sys.version_info >= (3, 7):
        # We need to give up on typing in Python 3.6
        assert isinstance(old_env, LegacyEnv)
    env = EnvCompatibility(old_env, render_mode="rgb_array")
    assert env.observation_space == Discrete(1)
    assert env.action_space == Discrete(1)
    assert env.reset() == (0, {})
    assert env.reset(seed=0, options={"some": "option"}) == (0, {})
    assert env.step(0) == (0, 0, False, False, {})
    assert env.render().shape == (1, 1, 3)
    env.close()


def test_make_compatibility_in_spec():
    gym.register(
        id="LegacyTestEnv-v0",
        entry_point=LegacyEnvExplicit,
        apply_api_compatibility=True,
    )
    env = gym.make("LegacyTestEnv-v0", render_mode="rgb_array")
    assert env.observation_space == Discrete(1)
    assert env.action_space == Discrete(1)
    assert env.reset() == (0, {})
    assert env.reset(seed=0, options={"some": "option"}) == (0, {})
    assert env.step(0) == (0, 0, False, False, {})
    img = env.render()
    assert isinstance(img, np.ndarray)
    assert img.shape == (1, 1, 3)  # type: ignore
    env.close()
    del gym.envs.registration.registry["LegacyTestEnv-v0"]


def test_make_compatibility_in_make():
    gym.register(id="LegacyTestEnv-v0", entry_point=LegacyEnvExplicit)
    env = gym.make(
        "LegacyTestEnv-v0", apply_api_compatibility=True, render_mode="rgb_array"
    )
    assert env.observation_space == Discrete(1)
    assert env.action_space == Discrete(1)
    assert env.reset() == (0, {})
    assert env.reset(seed=0, options={"some": "option"}) == (0, {})
    assert env.step(0) == (0, 0, False, False, {})
    img = env.render()
    assert isinstance(img, np.ndarray)
    assert img.shape == (1, 1, 3)  # type: ignore
    env.close()
    del gym.envs.registration.registry["LegacyTestEnv-v0"]
