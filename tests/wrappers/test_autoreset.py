from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

import gym
from gym.wrappers import AutoResetWrapper
from tests.envs.spec_list import spec_list


class DummyResetEnv(gym.Env):
    """
    A dummy environment which returns ascending numbers starting
    at 0 when self.step() is called. After the third call to self.step()
    done is true. Info dicts are also returned containing the same number
    returned as an observation, accessible via the key "count".
    This environment is provided for the purpose of testing the
    autoreset wrapper.
    """

    metadata = {}

    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float64
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0])
        )
        self.count = 0

    def step(self, action):
        self.count += 1
        return (
            np.array([self.count]),
            1 if self.count > 2 else 0,
            False,
            self.count > 2,
            {"count": self.count},
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: Optional[bool] = False,
        options: Optional[dict] = None
    ):
        self.count = 0
        if not return_info:
            return np.array([self.count])
        else:
            return np.array([self.count]), {"count": self.count}


def test_autoreset_reset_info():
    env = gym.make("CartPole-v1", return_two_dones=True)
    env = AutoResetWrapper(env)
    ob_space = env.observation_space
    obs = env.reset()
    assert ob_space.contains(obs)
    obs = env.reset(return_info=False)
    assert ob_space.contains(obs)
    obs, info = env.reset(return_info=True)
    assert ob_space.contains(obs)
    assert isinstance(info, dict)
    env.close()


@pytest.mark.parametrize("spec", spec_list, ids=[spec.id for spec in spec_list])
def test_make_autoreset_true(spec):
    """
    Note: This test assumes that the outermost wrapper is AutoResetWrapper
    so if that is being changed in the future, this test will break and need
    to be updated.
    Note: This test assumes that all first-party environments will terminate in a finite
    amount of time with random actions, which is true as of the time of adding this test.
    """
    env = None
    with pytest.warns(None):
        env = spec.make(autoreset=True, return_two_dones=True)

    env.reset(seed=0)
    env.action_space.seed(0)

    env.unwrapped.reset = MagicMock(side_effect=env.unwrapped.reset)

    terminated = False
    truncated = False
    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert isinstance(env, AutoResetWrapper)
    assert env.unwrapped.reset.called
    env.close()


@pytest.mark.parametrize("spec", spec_list, ids=[spec.id for spec in spec_list])
def test_make_autoreset_false(spec):
    env = None
    with pytest.warns(None):
        env = spec.make(autoreset=False, return_two_dones=True)
    assert not isinstance(env, AutoResetWrapper)
    env.close()


@pytest.mark.parametrize("spec", spec_list, ids=[spec.id for spec in spec_list])
def test_make_autoreset_default_false(spec):
    env = None
    with pytest.warns(None):
        env = spec.make(return_two_dones=True)
    assert not isinstance(env, AutoResetWrapper)
    env.close()


def test_autoreset_autoreset():
    env = DummyResetEnv()
    env = AutoResetWrapper(env)
    obs, info = env.reset(return_info=True)
    assert obs == np.array([0])
    assert info == {"count": 0}
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([1])
    assert reward == 0
    assert terminated is False
    assert truncated is False
    assert info == {"count": 1}
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([2])
    assert terminated is False
    assert truncated is False
    assert reward == 0
    assert info == {"count": 2}
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([0])
    assert terminated is False
    assert truncated is True
    assert reward == 1
    assert info == {
        "count": 0,
        "closing_observation": np.array([3]),
        "closing_info": {"count": 3},
    }
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([1])
    assert reward == 0
    assert terminated is False
    assert truncated is False
    assert info == {"count": 1}
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([2])
    assert reward == 0
    assert terminated is False
    assert truncated is False
    assert info == {"count": 2}
    env.close()
