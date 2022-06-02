from typing import Optional

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf

import gym
from gym.wrappers import jax_to_tf_v0


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


def test_tf_wrapper_reset_info():
    env = DummyJaxEnv(return_reward_idx=0)
    env = jax_to_tf_v0(env)
    obs = env.reset()
    assert isinstance(obs, tf.Tensor)
    del obs

    obs = env.reset(return_info=False)
    assert isinstance(obs, tf.Tensor)
    del obs

    obs, info = env.reset(return_info=True)
    assert isinstance(obs, tf.Tensor)
    assert isinstance(info, dict)


def test_tf_wrapper_step():
    env = DummyJaxEnv(return_reward_idx=0)
    env = jax_to_tf_v0(env)

    (obs, *_) = env.step(tf.convert_to_tensor(env.action_space.sample()))
    assert isinstance(obs, tf.Tensor)
    assert env.observation_space.contains(obs.numpy())


@pytest.mark.xfail(reason="VectorEnv for brax not yet implemented")
def test_tf_wrapper_vector_env():
    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs.reset()

    (obs, *_) = envs.step(tf.convert_to_tensor(envs.action_space.sample()))
    assert all(isinstance(o, tf.Tensor) for o in obs)
