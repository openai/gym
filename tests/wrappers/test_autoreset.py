"""Tests the gym.wrapper.AutoResetWrapper operates as expected."""

from typing import Generator, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

import gym
from gym.wrappers import AutoResetWrapper
from tests.envs.utils import all_testing_env_specs


class DummyResetEnv(gym.Env):
    """A dummy environment which returns ascending numbers starting at `0` when :meth:`self.step()` is called.

    After the second call to :meth:`self.step()` done is true.
    Info dicts are also returned containing the same number returned as an observation, accessible via the key "count".
    This environment is provided for the purpose of testing the autoreset wrapper.
    """

    metadata = {}

    def __init__(self):
        """Initialise the DummyResetEnv."""
        self.action_space = gym.spaces.Box(
            low=np.array([0]), high=np.array([2]), dtype=np.int64
        )
        self.observation_space = gym.spaces.Discrete(2)
        self.count = 0

    def step(self, action: int):
        """Steps the DummyEnv with the incremented step, reward and done `if self.count > 1` and updated info."""
        self.count += 1
        return (
            np.array([self.count]),  # Obs
            self.count > 2,  # Reward
            self.count > 2,  # Done
            {"count": self.count},  # Info
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: Optional[bool] = False,
        options: Optional[dict] = None
    ):
        """Resets the DummyEnv to return the count array and info with count."""
        self.count = 0
        if not return_info:
            return np.array([self.count])
        else:
            return np.array([self.count]), {"count": self.count}


def unwrap_env(env) -> Generator[gym.Wrapper, None, None]:
    """Unwraps an environment yielding all wrappers around environment."""
    while isinstance(env, gym.Wrapper):
        yield type(env)
        env = env.env


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_make_autoreset_true(spec):
    """Tests gym.make with `autoreset=True`, and check that the reset actually happens.

    Note: This test assumes that the outermost wrapper is AutoResetWrapper so if that
     is being changed in the future, this test will break and need to be updated.
    Note: This test assumes that all first-party environments will terminate in a finite
     amount of time with random actions, which is true as of the time of adding this test.
    """
    with pytest.warns(None):
        env = gym.make(spec.id, autoreset=True, disable_env_checker=True)
    assert AutoResetWrapper in unwrap_env(env)

    env.reset(seed=0)
    env.unwrapped.reset = MagicMock(side_effect=env.unwrapped.reset)

    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())

    assert env.unwrapped.reset.called
    env.close()


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_gym_make_autoreset(spec):
    """Tests that `gym.make` autoreset wrapper is applied only when `gym.make(..., autoreset=True)`."""
    with pytest.warns(None):
        env = gym.make(spec.id, disable_env_checker=True)
    assert AutoResetWrapper not in unwrap_env(env)
    env.close()

    with pytest.warns(None):
        env = gym.make(spec.id, autoreset=False, disable_env_checker=True)
    assert AutoResetWrapper not in unwrap_env(env)
    env.close()

    with pytest.warns(None):
        env = gym.make(spec.id, autoreset=True, disable_env_checker=True)
    assert AutoResetWrapper in unwrap_env(env)
    env.close()


def test_autoreset_wrapper_autoreset():
    """Tests the autoreset wrapper actually automatically resets correctly."""
    env = DummyResetEnv()
    env = AutoResetWrapper(env)

    obs, info = env.reset(return_info=True)
    assert obs == np.array([0])
    assert info == {"count": 0}

    action = 0
    obs, reward, done, info = env.step(action)
    assert obs == np.array([1])
    assert reward == 0
    assert done is False
    assert info == {"count": 1}

    obs, reward, done, info = env.step(action)
    assert obs == np.array([2])
    assert done is False
    assert reward == 0
    assert info == {"count": 2}

    obs, reward, done, info = env.step(action)
    assert obs == np.array([0])
    assert done is True
    assert reward == 1
    assert info == {
        "count": 0,
        "final_observation": np.array([3]),
        "final_info": {"count": 3},
        "TimeLimit.truncated": False,
    }

    obs, reward, done, info = env.step(action)
    assert obs == np.array([1])
    assert reward == 0
    assert done is False
    assert info == {"count": 1}

    env.close()
