from typing import Optional

import jax.numpy as jnp
import numpy as np
import pytest

import gym
from gym.wrappers import JaxToNumpyV0

class DummyJaxEnv(gym.Env):
    metadata = {}

    def __init__(self, return_reward_idx=0):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float64
        )
        self.returned_rewards = [0, 1, 2, 3, 4]
        self.return_reward_idx = return_reward_idx
        self.t = self.return_reward_idx

    def step(self, action):
        self.t += 1
        return jnp.array([self.t]), self.t, self.t == len(self.returned_rewards), {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: Optional[bool] = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self.t = self.return_reward_idx
        if not return_info:
            return jnp.array([self.t])
        else:
            return jnp.array([self.t]), {}

def make_env(return_reward_idx):
    def thunk():
        env = DummyJaxEnv(return_reward_idx)
        return env

    return thunk


def test_np_wrapper_reset_info():
    env = DummyJaxEnv(return_reward_idx=0)
    obs = env.reset()
    assert isinstance(obs, jnp.ndarray) and not isinstance(obs, np.ndarray)

    env = JaxToNumpyV0(env)
    np_obs = env.reset()
    assert isinstance(np_obs, np.ndarray)
    assert np_obs == jnp.asarray(obs)

    np_obs = env.reset(return_info=False)
    assert isinstance(np_obs, np.ndarray)
    assert np_obs == jnp.asarray(obs)

    np_obs, info = env.reset(return_info=True)
    assert isinstance(np_obs, np.ndarray)
    assert isinstance(info, dict)
    assert np_obs == jnp.asarray(obs)


def test_np_wrapper_step():
    env = DummyJaxEnv(return_reward_idx=0)
    env = JaxToNumpyV0(env)

    (obs, *_) = env.step(env.action_space.sample())
    assert isinstance(obs, np.ndarray)
    assert env.observation_space.contains(obs)


@pytest.mark.xfail(reason="VectorEnv for brax not yet implemented")
def test_np_wrapper_vector_env():
    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs.reset()

    (obs, *_) = envs.step(envs.action_space.sample())
    assert all(isinstance(o, np.ndarray) for o in obs)
