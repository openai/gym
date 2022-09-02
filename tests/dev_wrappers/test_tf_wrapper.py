from typing import Optional

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf

import gym
from gym.dev_wrappers.to_numpy import numpy_to_jax
from gym.dev_wrappers.to_tf import jax_to_tf, tf_to_jax
from gym.utils.env_checker import data_equivalence
from gym.vector import AsyncVectorEnv
from gym.wrappers import JaxToTFV0
from tests.dev_wrappers.utils import is_same_types
from tests.testing_env import GenericTestEnv


def _torch_equivalence(data_1, data_2):
    if type(data_1) == type(data_2):
        if isinstance(data_1, dict):
            return data_1.keys() == data_2.keys() and all(
                data_equivalence(data_1[k], data_2[k]) for k in data_1.keys()
            )
        elif isinstance(data_1, tuple):
            return len(data_1) == len(data_2) and all(
                data_equivalence(o_1, o_2) for o_1, o_2 in zip(data_1, data_2)
            )
        elif isinstance(data_1, tf.Tensor):
            return tf.math.reduce_all(data_1 == data_2)
        else:
            return data_1 == data_2
    else:
        return False


@pytest.mark.parametrize(
    "value",
    [
        tf.convert_to_tensor(1),
        tf.convert_to_tensor(2, dtype=tf.int8),
        tf.convert_to_tensor(3.1),
        tf.convert_to_tensor(2.7, dtype=tf.float16),
        tf.convert_to_tensor([1, 2, 3]),
        tf.convert_to_tensor([4, 5, 6], dtype=tf.int16),
        tuple(tf.convert_to_tensor(i) for i in range(3)),
        tuple(tf.convert_to_tensor(i, dtype=tf.float64) for i in range(3, 6)),
        {f"{i}": tf.convert_to_tensor(i) for i in range(3)},
        {f"{i}": tf.convert_to_tensor(i, dtype=tf.float32) for i in range(3, 6)},
        (tf.convert_to_tensor(0), (tf.convert_to_tensor(1), tf.convert_to_tensor(2))),
        (
            tf.convert_to_tensor(3),
            {"a": tf.convert_to_tensor(3), "b": tf.convert_to_tensor(4)},
        ),
        {
            "a": tf.convert_to_tensor(3.2),
            "b": {"c": tf.convert_to_tensor(4.3), "d": tf.convert_to_tensor(7.8)},
        },
        {
            "a": tf.convert_to_tensor(1.2),
            "b": (tf.convert_to_tensor(0), tf.convert_to_tensor(1.2)),
        },
        # Todo, in the future add graph instances
    ],
)
def test_roundtripping(value):
    assert is_same_types(tf_to_jax(value), jnp.DeviceArray)
    assert _torch_equivalence(jax_to_tf(tf_to_jax(value)), value)


def testing_jax_env_step(self, action):
    assert is_same_types(action, jnp.DeviceArray)
    return (
        numpy_to_jax(self.observation_space.sample()),
        jnp.array(0),
        jnp.array(False),
        jnp.array(True),
        {"Test-info": jnp.array([0, 1, 2])},
    )


TESTING_JAX_ENV_GEN = lambda obs_space, action_space: GenericTestEnv(
    observation_space=obs_space,
    action_space=action_space,
    reset_fn=lambda self, *_: numpy_to_jax(self.action_space.sample()),
    step_fn=testing_jax_env_step,
)


@pytest.mark.parametrize(
    "env",
    [
        # todo - add a couple of 3/4 of environments with variety of obs and action space
        TESTING_JAX_ENV_GEN(gym.spaces.Discrete(5), gym.spaces.Discrete(2)),
        AsyncVectorEnv(
            [
                lambda: TESTING_JAX_ENV_GEN(
                    gym.spaces.Discrete(5), gym.spaces.Discrete(2)
                )
                for _ in range(3)
            ],
            new_step_api=True,
        ),
    ],
)
def test_wrapper(env):
    wrapped_env = JaxToTFV0(env)

    returns = wrapped_env.reset()
    assert is_same_types(returns, tf.Tensor)

    # As we don't have jax spaces, we are converting to jax from numpy quickly
    action = jax_to_tf(numpy_to_jax(env.action_space.sample()))
    returns = wrapped_env.step(action)
    assert is_same_types(returns, tf.Tensor)


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
        options: Optional[dict] = None,
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
    obs = env.reset()
    assert isinstance(obs, jnp.ndarray) and not isinstance(obs, np.ndarray)

    env = JaxToTFV0(env)
    tf_obs = env.reset()
    assert isinstance(tf_obs, tf.Tensor)
    assert tf_obs.numpy() == jnp.asarray(obs)

    tf_obs = env.reset(return_info=False)
    assert isinstance(tf_obs, tf.Tensor)
    assert tf_obs.numpy() == jnp.asarray(obs)

    tf_obs, info = env.reset(return_info=True)
    assert isinstance(tf_obs, tf.Tensor)
    assert isinstance(info, dict)
    assert tf_obs.numpy() == jnp.asarray(obs)


def test_tf_wrapper_step():
    env = DummyJaxEnv(return_reward_idx=0)
    env = JaxToTFV0(env)

    (obs, *_) = env.step(tf.convert_to_tensor(env.action_space.sample()))
    assert isinstance(obs, tf.Tensor)
    assert env.observation_space.contains(obs.numpy())
