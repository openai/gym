"""Tests for the filter observation wrapper."""


import pytest
import numpy as np

import gym
from gym.spaces import Dict, Box, Discrete, Tuple
from gym.wrappers.dict import FlattenDictWrapper


class FakeEnvironment(gym.Env):
    def __init__(self, observation_space):
        self.observation_space = observation_space
        self.obs_keys = self.observation_space.spaces.keys()
        self.action_space = Box(
            shape=(1, ), low=-1, high=1, dtype=np.float32)

    def render(self, width=32, height=32, *args, **kwargs):
        del args
        del kwargs
        image_shape = (height, width, 3)
        return np.zeros(image_shape, dtype=np.uint8)

    def reset(self):
        observation = self.observation_space.sample()
        return observation

    def step(self, action):
        del action
        observation = self.observation_space.sample()
        reward, terminal, info = 0.0, False, {}
        return observation, reward, terminal, info


NESTED_DICT_TEST_CASES = (
     (Dict({
        "key1": Box(shape=(2, ), low=-1, high=1, dtype=np.float32),
        "key2": Dict({
            "subkey1": Box(shape=(2, ), low=-1, high=1, dtype=np.float32),
            "subkey2": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
        })
     }), (6, )),
     (Dict({
         "key1": Box(shape=(2, 3), low=-1, high=1, dtype=np.float32),
         "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
         "key3": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
     }), (9, )),
     (Dict({
         "key1": Tuple(
             (Box(shape=(2, ), low=-1, high=1, dtype=np.float32),
              Box(shape=(2, ), low=-1, high=1, dtype=np.float32))
         ),
         "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
         "key3": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
     }), (7, )),
     (Dict({
         "key1": Tuple(
             (Box(shape=(2, ), low=-1, high=1, dtype=np.float32), )
         ),
         "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
         "key3": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
     }), (5, )),
     (Dict({
         "key1": Tuple(
             (Dict({
                 "key9": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
             }), )
         ),
         "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
         "key3": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
     }), (5, )),
)

ERROR_TEST_CASES = (
    (Dict({
        "key1": Box(shape=(2, ), low=-1, high=1, dtype=np.float32),
        "key2": Dict({
            "subkey1": Discrete(n=2),
            "subkey2": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
        })
    }), AssertionError, "Only spaces of type Box are supported."),
    (Dict({
        "key2": Discrete(n=2),
        "key3": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
    }), AssertionError, "Only spaces of type Box are supported."),
    (Dict({
        "key1": Tuple(
            (Dict({
                "key1": Discrete(n=2)
            }), )
        ),
        "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
        "key3": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
    }), AssertionError, "Only spaces of type Box are supported."),
    (Dict({
        "key1": Box(shape=(2, ), low=-1, high=1, dtype=np.float32),
        "key2": Dict({
            "subkey1": Box(shape=(2, ), low=-1, high=1, dtype=np.float64)
        })
    }), AssertionError, "All spaces must have the same dtype."),
    (Dict({
        "key1": Tuple(
            (Dict({

                "subkey1": Box(shape=(), low=-1, high=1, dtype=np.float32),
                "subkey2": Box(shape=(), low=-1, high=1, dtype=np.float32),
                "subkey3": Box(shape=(), low=-1, high=1, dtype=np.int64),
                "subkey4": Box(shape=(), low=-1, high=1, dtype=np.float32),
            }), )
        ),
        "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
        "key3": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
    }), AssertionError, "All spaces must have the same dtype."),
    (Dict({
        "key2": Box(shape=(), low=-1, high=1, dtype=np.int32),
        "key3": Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
    }), AssertionError, "All spaces must have the same dtype."),
)


class TestNestedDictWrapper(object):
    @pytest.mark.parametrize("observation_space, flat_shape",

                             NESTED_DICT_TEST_CASES)
    def test_nested_dicts_size(self, observation_space, flat_shape):
        env = FakeEnvironment(observation_space=observation_space)

        # Make sure we are testing the right environment for the test.
        observation_space = env.observation_space
        assert isinstance(observation_space, Dict)

        wrapped_env = FlattenDictWrapper(env, env.obs_keys)
        assert wrapped_env.observation_space.shape == flat_shape

        assert wrapped_env.observation_space.dtype == wrapped_env.dtype

    @pytest.mark.parametrize("observation_space, flat_shape", NESTED_DICT_TEST_CASES)
    def test_nested_dicts_ravel(self, observation_space, flat_shape):
        env = FakeEnvironment(observation_space=observation_space)
        wrapped_env = FlattenDictWrapper(env, env.obs_keys)
        obs = wrapped_env.reset()
        assert obs.shape == wrapped_env.observation_space.shape

    @pytest.mark.parametrize("observation_space,error_type,error_match",
                             ERROR_TEST_CASES)
    def test_raises_with_incorrect_arguments(self,
                                             observation_space,
                                             error_type,
                                             error_match):
        env = FakeEnvironment(observation_space=observation_space)

        with pytest.raises(error_type, match=error_match):
            FlattenDictWrapper(env, env.obs_keys)
