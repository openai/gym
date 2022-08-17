from typing import Optional

import jax.numpy as jnp
import numpy as np
import pytest
import torch

import gym
from gym.dev_wrappers.torch_wrapper import jax_to_torch, torch_to_jax
from gym.wrappers import JaxToTorchV0


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
    obs = env.reset()
    assert isinstance(obs, jnp.ndarray) and not isinstance(obs, np.ndarray)

    env = JaxToTorchV0(env)
    torch_obs = env.reset()
    assert isinstance(torch_obs, torch.Tensor)
    assert torch_obs.numpy() == jnp.asarray(obs)

    torch_obs = env.reset(return_info=False)
    assert isinstance(torch_obs, torch.Tensor)
    assert torch_obs.numpy() == jnp.asarray(obs)

    torch_obs, info = env.reset(return_info=True)
    assert isinstance(torch_obs, torch.Tensor)
    assert isinstance(info, dict)
    assert torch_obs.numpy() == jnp.asarray(obs)


def test_torch_wrapper_step():
    env = DummyJaxEnv(return_reward_idx=0)
    env = JaxToTorchV0(env)

    (obs, *_) = env.step(torch.Tensor(env.action_space.sample()))
    assert isinstance(obs, torch.Tensor)
    assert env.observation_space.contains(obs.numpy())


def test_torch_wrapper_device():
    env = DummyJaxEnv(return_reward_idx=0)
    env = JaxToTorchV0(env, device="meta")

    (obs, *_) = env.step(torch.Tensor(env.action_space.sample()))
    assert isinstance(obs, torch.Tensor)
    assert obs.device == torch.device("meta")


@pytest.mark.xfail(reason="VectorEnv for brax not yet implemented")
def test_torch_wrapper_vector_env():
    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs.reset()

    (obs, *_) = envs.step(torch.Tensor(envs.action_space.sample()))
    assert all(isinstance(o, torch.Tensor) for o in obs)


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        2,
        (3.0, 4.0, 5.0),
        [3.0, 4.0, 5.0],
        {
            "a": 6.0,
            "b": 7,
        },
    ],
)
def test_torch_conversion(value):
    assert torch_to_jax(jax_to_torch(value)) == value


@pytest.mark.parametrize(
    "value", [np.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0])]
)
def test_torch_array_conversion(value):
    assert (torch_to_jax(jax_to_torch(value)) == value).all()
