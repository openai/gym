import gym
import numpy as np
from numpy.testing import assert_almost_equal

from gym.wrappers.normalize import NormalizeObservation, NormalizeReward


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


def test_normalize_observation():
    env = DummyRewardEnv(return_reward_idx=0)
    env = NormalizeObservation(env)
    env.reset()
    env.step(env.action_space.sample())
    assert_almost_equal(env.obs_rms.mean, 0.5, decimal=4)
    env.step(env.action_space.sample())
    assert_almost_equal(env.obs_rms.mean, 1.0, decimal=4)


def test_normalize_return():
    env = DummyRewardEnv(return_reward_idx=0)
    env = NormalizeReward(env)
    env.reset()
    env.step(env.action_space.sample())
    assert_almost_equal(
        env.return_rms.mean,
        np.mean([1]),  # [first return]
        decimal=4,
    )
    env.step(env.action_space.sample())
    assert_almost_equal(
        env.return_rms.mean,
        np.mean([2 + env.gamma * 1, 1]),  # [second return, first return]
        decimal=4,
    )


def test_normalize_observation_vector_env():
    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs.reset()
    obs, reward, _, _ = envs.step(envs.action_space.sample())
    np.testing.assert_almost_equal(obs, np.array([[1], [2]]), decimal=4)
    np.testing.assert_almost_equal(reward, np.array([1, 2]), decimal=4)

    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs = NormalizeObservation(envs)
    envs.reset()
    assert_almost_equal(
        envs.obs_rms.mean,
        np.mean([0.5]),  # the mean of first observations [[0, 1]]
        decimal=4,
    )
    obs, reward, _, _ = envs.step(envs.action_space.sample())
    assert_almost_equal(
        envs.obs_rms.mean,
        np.mean([1.0]),  # the mean of first and second observations [[0, 1], [1, 2]]
        decimal=4,
    )


def test_normalize_return_vector_env():
    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs = NormalizeReward(envs)
    obs = envs.reset()
    obs, reward, _, _ = envs.step(envs.action_space.sample())
    assert_almost_equal(
        envs.return_rms.mean,
        np.mean([1.5]),  # the mean of first returns [[1, 2]]
        decimal=4,
    )
    obs, reward, _, _ = envs.step(envs.action_space.sample())
    assert_almost_equal(
        envs.return_rms.mean,
        np.mean(
            [[1, 2], [2 + envs.gamma * 1, 3 + envs.gamma * 2]]
        ),  # the mean of first and second returns [[1, 2], [2 + envs.gamma * 1, 3 + envs.gamma * 2]]
        decimal=4,
    )
