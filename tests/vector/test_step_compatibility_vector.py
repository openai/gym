import numpy as np
import pytest

import gym
from gym.spaces import Discrete
from gym.vector import AsyncVectorEnv, SyncVectorEnv


class OldStepEnv(gym.Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(2)

    def reset(self):
        return 0

    def step(self, action):
        obs = self.observation_space.sample()
        rew = 0
        done = False
        info = {}
        return obs, rew, done, info


class NewStepEnv(gym.Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(2)

    def reset(self):
        return 0

    def step(self, action):
        obs = self.observation_space.sample()
        rew = 0
        terminated = False
        truncated = False
        info = {}
        return obs, rew, terminated, truncated, info


@pytest.mark.parametrize("VecEnv", [AsyncVectorEnv, SyncVectorEnv])
def test_vector_step_compatibility_new_env(VecEnv):

    envs = [
        OldStepEnv(),
        NewStepEnv(),
    ]

    vec_env = VecEnv([lambda: env for env in envs])
    vec_env.reset()
    step_returns = vec_env.step([0, 0])
    assert len(step_returns) == 4
    _, _, dones, _ = step_returns
    assert dones.dtype == np.bool_
    vec_env.close()

    vec_env = VecEnv([lambda: env for env in envs], new_step_api=True)
    vec_env.reset()
    step_returns = vec_env.step([0, 0])
    assert len(step_returns) == 5
    _, _, terminateds, truncateds, _ = step_returns
    assert terminateds.dtype == np.bool_
    assert truncateds.dtype == np.bool_
    vec_env.close()


@pytest.mark.parametrize("async_bool", [True, False])
def test_vector_step_compatibility_existing(async_bool):

    env = gym.vector.make("CartPole-v1", num_envs=3, asynchronous=async_bool)
    env.reset()
    step_returns = env.step(env.action_space.sample())
    assert len(step_returns) == 4
    _, _, dones, _ = step_returns
    assert dones.dtype == np.bool_
    env.close()

    env = gym.vector.make(
        "CartPole-v1", num_envs=3, asynchronous=async_bool, new_step_api=True
    )
    env.reset()
    step_returns = env.step(env.action_space.sample())
    assert len(step_returns) == 5
    _, _, terminateds, truncateds, _ = step_returns
    assert terminateds.dtype == np.bool_
    assert truncateds.dtype == np.bool_
    env.close()
