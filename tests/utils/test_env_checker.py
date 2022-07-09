from typing import Optional

import numpy as np
import pytest

import gym
from gym.spaces import Box, Dict, Discrete
from gym.utils.env_checker import check_env


class ActionDictTestEnv(gym.Env):
    action_space = Dict({"position": Discrete(1), "velocity": Discrete(1)})
    observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)

    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode

    def step(self, action):
        observation = np.array([1.0, 1.5, 0.5])
        reward = 1
        done = True
        return observation, reward, done

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        return np.array([1.0, 1.5, 0.5])

    def render(self, mode: Optional[str] = "human"):
        pass


def test_check_env_dict_action():
    # Environment.step() only returns 3 values: obs, reward, done. Not info!
    test_env = ActionDictTestEnv()

    with pytest.raises(AssertionError) as errorinfo:
        check_env(env=test_env, warn=True)
        assert (
            str(errorinfo.value)
            == "The `step()` method must return four values: obs, reward, done, info"
        )


def test_check_env_dict_observation():
    """Tests passive checking for observations that are dictionaries.

    The test environment returns observations that are dictionaries of different
    types. Then three tests are performed: The first test checks if all passive
    checks are passing. The second inserts a nan value into the observation
    dictionary and tests for a failing passive check. Then the same test is
    repeated by inserting an inf value into the observation dictionary.
    """

    class TestDictEnv(gym.Env):
        def __init__(self, test_value=None):
            self.actions = [0]
            self.action_space = Discrete(len(self.actions))

            self.observation_space = Dict(
                {
                    "img": Box(low=0, high=255, shape=(8, 8, 3), dtype="uint8"),
                    "pos": Box(low=0.0, high=1.0, shape=(2,), dtype="float32"),
                    "idx": Discrete(4),
                }
            )
            self.reward_range = (0, 1)
            self._test_value = test_value

        def _get_observation(self):
            return {
                "img": np.zeros((8, 8, 3), dtype=np.uint8),
                "pos": np.zeros(2, dtype=np.float32),
                "idx": 0,
            }

        def reset(self, seed=None, return_info=None, options=None):
            return self._get_observation()

        def step(self, action):
            observation = self._get_observation()
            if self._test_value is not None:
                observation["pos"][0] = self._test_value
            return observation, 0.0, False, False, {}

    from gym.wrappers.env_checker import PassiveEnvChecker

    env = TestDictEnv()
    env = PassiveEnvChecker(env)

    obs = env.reset()
    assert isinstance(obs, dict)
    assert isinstance(obs["img"], np.ndarray)
    assert isinstance(obs["pos"], np.ndarray)
    assert isinstance(obs["idx"], int)

    obs, rew, done, truncated, info = env.step(0)
    assert isinstance(obs, dict)
    assert isinstance(obs["img"], np.ndarray)
    assert isinstance(obs["pos"], np.ndarray)
    assert isinstance(obs["idx"], int)
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env = TestDictEnv(test_value=np.nan)
    env = PassiveEnvChecker(env)
    try:
        obs, rew, done, truncated, info = env.step(0)
        pytest.fail()
    except AssertionError:
        pass

    env = TestDictEnv(test_value=np.inf)
    env = PassiveEnvChecker(env)
    try:
        obs, rew, done, truncated, info = env.step(0)
        pytest.fail()
    except AssertionError:
        pass
