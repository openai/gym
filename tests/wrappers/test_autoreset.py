import pytest
from typing import Optional

import numpy as np

import gym
from gym.wrappers import AutoResetWrapper


class DummyResetEnv(gym.Env):
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
    del obs
    obs = env.reset(return_info=False)
    assert ob_space.contains(obs)
    del obs
    obs, info = env.reset(return_info=True)
    assert ob_space.contains(obs)
    assert isinstance(info, dict)


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
    assert info == {"info": {"count": 1}}
    obs, reward, done, info = env.step(action)
    assert obs == np.array([2])
    assert done == False
    assert reward == 0
    assert info == {"info": {"count": 2}}
    obs, reward, done, info = env.step(action)
    assert obs == np.array([0])
    assert done == True
    assert reward == 1
    assert info == {
        "info": {"count": 0},
        "final_obs": np.array([3]),
        "final_info": {"count": 3},
    }
    obs, reward, done, info = env.step(action)
    assert obs == np.array([1])
    assert reward == 0
    assert done == False
    assert info == {"info": {"count": 1}}
    obs, reward, done, info = env.step(action)
    assert obs == np.array([2])
    assert reward == 0
    assert done == False
    assert info == {"info": {"count": 2}}
