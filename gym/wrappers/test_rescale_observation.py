import pytest

import numpy as np

import gym
from gym import spaces
from gym.wrappers import RescaleObservation


class FakeEnvironment(gym.Env):
    def __init__(self):
        """Fake environment whose observation equals broadcasted action."""
        self.observation_space = gym.spaces.Box(
            shape=(2, ),
            low=np.array((-1.2, -0.07)),
            high=np.array((0.6, 0.07)),
            dtype=np.float32)
        self.action_space = self.observation_space

    def reset(self):
        observation = self.observation_space.sample()
        return observation

    def step(self, action):
        observation = action * np.ones(self.observation_space.shape)
        reward, terminal, info = 0.0, False, {}
        return observation, reward, terminal, info


def test_rescale_observation():
    new_low, new_high = -1.0, 1.0
    env = FakeEnvironment()
    wrapped_env = RescaleObservation(env, new_low, new_high)

    np.testing.assert_allclose(wrapped_env.observation_space.low, new_low)
    np.testing.assert_allclose(wrapped_env.observation_space.high, new_high)

    seed = 0
    env.seed(seed)
    wrapped_env.seed(seed)

    env.reset()
    wrapped_env.reset()

    low_observation = env.step(env.observation_space.low)[0]
    wrapped_low_observation = wrapped_env.step(env.observation_space.low)[0]

    assert np.allclose(low_observation, env.observation_space.low)
    assert np.allclose(
        wrapped_low_observation, wrapped_env.observation_space.low)

    high_observation = env.step(env.observation_space.high)[0]
    wrapped_high_observation = wrapped_env.step(env.observation_space.high)[0]

    assert np.allclose(high_observation, env.observation_space.high)
    assert np.allclose(
        wrapped_high_observation, wrapped_env.observation_space.high)


def test_raises_on_non_finite_low():
    env = FakeEnvironment()
    assert isinstance(env.observation_space, spaces.Box)

    with pytest.raises(ValueError):
        RescaleObservation(env, -float('inf'), 1.0)

    with pytest.raises(ValueError):
        RescaleObservation(env, -1.0, float('inf'))

    with pytest.raises(ValueError):
        RescaleObservation(env, -1.0, np.nan)


def test_raises_on_high_less_than_low():
    env = FakeEnvironment()
    assert isinstance(env.observation_space, spaces.Box)
    with pytest.raises(ValueError):
        RescaleObservation(env, 1.0, 1.0)
    with pytest.raises(ValueError):
        RescaleObservation(env, 1.0, -1.0)


def test_raises_on_high_equals_low():
    env = FakeEnvironment()
    assert isinstance(env.observation_space, spaces.Box)
    with pytest.raises(ValueError):
        RescaleObservation(env, 1.0, 1.0)


def test_raises_on_non_box_space():
    env = gym.envs.make('Copy-v0')
    assert isinstance(env.observation_space, spaces.Discrete)
    with pytest.raises(TypeError):
        RescaleObservation(env, -1.0, 1.0)


def test_raises_on_non_finite_space():
    env = gym.envs.make('Swimmer-v3')
    assert np.any(np.isinf((
        env.observation_space.low, env.observation_space.high)))
    with pytest.raises(ValueError):
        RescaleObservation(env, -1.0, 1.0)
