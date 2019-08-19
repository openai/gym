import pytest
import numpy as np

import gym
from gym import spaces
from gym.wrappers.filter_observation import FilterObservation


class FakeEnvironment(gym.Env):
    def __init__(self, observation_keys=('state')):
        self.observation_space = spaces.Dict({
            name: spaces.Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
            for name in observation_keys
        })
        self.action_space = spaces.Box(
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


FILTER_OBSERVATION_TEST_CASES = (
    (('key1', 'key2'), ('key1', )),
    (('key1', 'key2'), ('key1', 'key2')),
    (('key1', ), None),
    (('key1', ), ('key1', )),
)

ERROR_TEST_CASES = (
    ('key', ValueError, "All the filter_keys must be included..*"),
    (False, TypeError, "'bool' object is not iterable"),
    (1, TypeError, "'int' object is not iterable"),
)


class TestFilterObservation(object):
    @pytest.mark.parametrize("observation_keys,filter_keys",
                             FILTER_OBSERVATION_TEST_CASES)
    def test_filter_observation(self, observation_keys, filter_keys):
        env = FakeEnvironment(observation_keys=observation_keys)

        # Make sure we are testing the right environment for the test.
        observation_space = env.observation_space
        assert isinstance(observation_space, spaces.Dict)

        wrapped_env = FilterObservation(env, filter_keys=filter_keys)

        assert isinstance(wrapped_env.observation_space, spaces.Dict)

        if filter_keys is None:
            filter_keys = tuple(observation_keys)

        assert len(wrapped_env.observation_space.spaces) == len(filter_keys)
        assert (tuple(wrapped_env.observation_space.spaces.keys())
                == tuple(filter_keys))

        # Check that the added space item is consistent with the added observation.
        observation = wrapped_env.reset()
        assert (len(observation) == len(filter_keys))

    @pytest.mark.parametrize("filter_keys,error_type,error_match",
                             ERROR_TEST_CASES)
    def test_raises_with_incorrect_arguments(self,
                                             filter_keys,
                                             error_type,
                                             error_match):
        env = FakeEnvironment(observation_keys=('key1', 'key2'))

        ValueError

        with pytest.raises(error_type, match=error_match):
            FilterObservation(env, filter_keys=filter_keys)
