from typing import Optional

import jax.numpy as jnp
import numpy as np
import pytest
import torch

import gym
from gym.dev_wrappers.torch_wrapper import TorchWrapper

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


def test_torch_wrapper_reset_info():
    env = DummyJaxEnv(return_reward_idx=0)
    env = TorchWrapper(env)
    obs = env.reset()
    assert isinstance(obs, torch.Tensor)
    del obs

    obs = env.reset(return_info=False)
    assert isinstance(obs, torch.Tensor)
    del obs

    obs, info = env.reset(return_info=True)
    assert isinstance(obs, torch.Tensor)
    assert isinstance(info, dict)


def test_torch_wrapper_step():
    env = DummyJaxEnv(return_reward_idx=0)
    env = TorchWrapper(env)

    (obs, *_) = env.step(torch.Tensor(env.action_space.sample()))
    assert isinstance(obs, torch.Tensor)
    assert env.observation_space.contains(obs.numpy())


def test_torch_wrapper_device():
    env = DummyJaxEnv(return_reward_idx=0)
    env = TorchWrapper(env, device="meta")

    (obs, *_) = env.step(torch.Tensor(env.action_space.sample()))
    assert isinstance(obs, torch.Tensor)
    assert obs.device == torch.device("meta")

