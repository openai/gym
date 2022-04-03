import types
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

import gym
from gym.wrappers import AutoResetWrapper


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
        self.action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]))
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0])
        )
        self.count = 0

    def step(self, action):
        self.count += 1
        return (
            np.array([self.count]),
            1 if self.count > 2 else 0,
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
    env = gym.make("CartPole-v1")
    env = AutoResetWrapper(env)
    ob_space = env.observation_space
    obs = env.reset()
    assert ob_space.contains(obs)
    obs = env.reset(return_info=False)
    assert ob_space.contains(obs)
    obs, info = env.reset(return_info=True)
    assert ob_space.contains(obs)
    assert isinstance(info, dict)


def test_make_autoreset():

    env = gym.make("CartPole-v1", autoreset=True)
    ob_space = env.observation_space
    obs = env.reset(seed=0)
    env.action_space.seed(0)

    env.env.reset = MagicMock(side_effect=env.env.reset)

    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())

    assert env.env.reset.called

    env = gym.make("CartPole-v1", autoreset=False)
    ob_space = env.observation_space
    obs = env.reset(seed=0)
    env.action_space.seed(0)

    env.env.reset = MagicMock(side_effect=env.env.reset)

    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
    assert not env.env.reset.called

    env = gym.make("CartPole-v1")
    ob_space = env.observation_space
    obs = env.reset(seed=0)
    env.action_space.seed(0)

    env.env.reset = MagicMock(side_effect=env.env.reset)

    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
    assert not env.env.reset.called


def test_autoreset_autoreset():
    env = DummyResetEnv()
    env = AutoResetWrapper(env)
    obs, info = env.reset(return_info=True)
    assert obs == np.array([0])
    assert info == {"count": 0}
    action = 1
    obs, reward, done, info = env.step(action)
    assert obs == np.array([1])
    assert reward == 0
    assert done == False
    assert info == {"count": 1}
    obs, reward, done, info = env.step(action)
    assert obs == np.array([2])
    assert done == False
    assert reward == 0
    assert info == {"count": 2}
    obs, reward, done, info = env.step(action)
    assert obs == np.array([0])
    assert done == True
    assert reward == 1
    assert info == {
        "count": 0,
        "terminal_observation": np.array([3]),
        "terminal_info": {"count": 3},
    }
    obs, reward, done, info = env.step(action)
    assert obs == np.array([1])
    assert reward == 0
    assert done == False
    assert info == {"count": 1}
    obs, reward, done, info = env.step(action)
    assert obs == np.array([2])
    assert reward == 0
    assert done == False
    assert info == {"count": 2}
