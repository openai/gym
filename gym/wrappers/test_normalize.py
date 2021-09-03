import gym
import numpy as np
from numpy.testing import assert_almost_equal

from gym.wrappers.normalize import Normalize


class DummyRewardEnv(gym.Env):
    metadata = {}

    def __init__(self, return_reward_idx=0):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0])
        )
        self.returned_rewards = [0, 1, 2, 3, 4]
        self.return_reward_idx = return_reward_idx
        self.t = self.return_reward_idx

    def step(self, action):
        self.t += 1
        return np.array([self.t]), self.t, self.t == len(self.returned_rewards), {}

    def reset(self):
        self.t = self.return_reward_idx
        return np.array([self.t])


def make_env(return_reward_idx):
    def thunk():
        env = DummyRewardEnv(return_reward_idx)
        return env

    return thunk


def test_normalize():
    env = DummyRewardEnv(return_reward_idx=0)
    env = Normalize(env)
    env.reset()
    env.step(env.action_space.sample())
    assert_almost_equal(0.5, env.obs_rms.mean, decimal=4)
    env.step(env.action_space.sample())
    assert_almost_equal(
        np.mean([2 + env.gamma * 1, 1]),  # [second return, first return]
        env.return_rms.mean,
        decimal=4,
    )


def test_normalize_vector_env():
    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs.reset()
    obs, reward, _, _ = envs.step(envs.action_space.sample())
    np.testing.assert_almost_equal(np.array([[1], [2]]), obs, decimal=4)
    np.testing.assert_almost_equal(np.array([1, 2]), reward, decimal=4)

    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs = Normalize(envs)
    envs.reset()
    assert_almost_equal(
        np.mean([0.5]),  # the mean of first observations [0, 1]
        envs.obs_rms.mean,
        decimal=4,
    )
    obs, reward, _, _ = envs.step(envs.action_space.sample())
    assert_almost_equal(
        np.mean([1.0]),  # the mean of first and second observations [[0, 1], [1, 2]]
        envs.obs_rms.mean,
        decimal=4,
    )
