import pytest

import numpy as np

import gym
from gym import spaces
from gym.wrappers import RescaleObservation


UNSCALED_BOX_SPACE = spaces.Box(
    shape=(2, ),
    low=np.array((-1.2, -0.07)),
    high=np.array((0.6, 0.07)),
    dtype=np.float32)


class FakeEnvironment(gym.Env):
    def __init__(self, observation_space):
        """Fake environment whose observation equals broadcasted action."""
        self.observation_space = observation_space
        self.action_space = self.observation_space

    def reset(self):
        observation = self.observation_space.sample()
        return observation

    def step(self, action):
        observation = action  # * np.ones(self.observation_space.shape)
        reward, terminal, info = 0.0, False, {}
        return observation, reward, terminal, info


@pytest.mark.parametrize("observation_space", [
    UNSCALED_BOX_SPACE,
    spaces.Tuple((UNSCALED_BOX_SPACE, UNSCALED_BOX_SPACE)),
    spaces.Dict({'box-1': UNSCALED_BOX_SPACE, 'box-2': UNSCALED_BOX_SPACE}),
])
def test_rescale_observation(observation_space):
    new_low, new_high = -1.0, 1.0
    env = FakeEnvironment(observation_space)
    wrapped_env = RescaleObservation(env, new_low, new_high)

    def verify_space_bounds(observation_space):
        np.testing.assert_allclose(observation_space.low, new_low)
        np.testing.assert_allclose(observation_space.high, new_high)

    if isinstance(wrapped_env.observation_space, spaces.Box):
        verify_space_bounds(wrapped_env.observation_space)
    elif isinstance(wrapped_env.observation_space, spaces.Tuple):
        for observation_space in wrapped_env.observation_space.spaces:
            verify_space_bounds(observation_space)
    elif isinstance(wrapped_env.observation_space, spaces.Dict):
        for observation_space in wrapped_env.observation_space.spaces.values():
            verify_space_bounds(observation_space)
    else:
        raise ValueError

    seed = 0
    env.seed(seed)
    wrapped_env.seed(seed)

    env.reset()
    wrapped_env.reset()

    if isinstance(wrapped_env.observation_space, spaces.Box):
        action = env.observation_space.low
        low_observation = env.step(action)[0]
        wrapped_low_observation = wrapped_env.step(action)[0]

        assert np.allclose(low_observation, env.observation_space.low)
        assert np.allclose(
            wrapped_low_observation, wrapped_env.observation_space.low)

        high_observation = env.step(env.observation_space.high)[0]
        wrapped_high_observation = wrapped_env.step(env.observation_space.high)[0]

        assert np.allclose(high_observation, env.observation_space.high)
        assert np.allclose(
            wrapped_high_observation, wrapped_env.observation_space.high)

    elif isinstance(wrapped_env.observation_space, spaces.Tuple):
        low_action = type(env.observation_space.spaces)(
            observation_space.low
            for observation_space in env.observation_space.spaces)

        low_observation = env.step(low_action)[0]
        wrapped_low_observation = wrapped_env.step(low_action)[0]

        assert np.allclose(
            low_observation,
            [o.low for o in env.observation_space.spaces])
        assert np.allclose(
            wrapped_low_observation,
            [o.low for o in wrapped_env.observation_space.spaces])

        high_action = type(env.observation_space.spaces)(
            observation_space.high
            for observation_space in env.observation_space.spaces)

        high_observation = env.step(high_action)[0]
        wrapped_high_observation = wrapped_env.step(high_action)[0]

        assert np.allclose(
            high_observation,
            [o.high for o in env.observation_space.spaces])
        assert np.allclose(
            wrapped_high_observation,
            [o.high for o in wrapped_env.observation_space.spaces])

    elif isinstance(wrapped_env.observation_space, spaces.Dict):
        low_action = type(env.observation_space.spaces)(
            (key, observation_space.low)
            for key, observation_space in env.observation_space.spaces.items())

        low_observation = env.step(low_action)[0]
        wrapped_low_observation = wrapped_env.step(low_action)[0]

        assert (set(env.observation_space.spaces.keys())
                == set(low_observation.keys()))
        assert (set(wrapped_env.observation_space.spaces.keys())
                == set(low_observation.keys()))
        for key in env.observation_space.spaces.keys():
            np.testing.assert_allclose(
                low_observation[key], env.observation_space[key].low)
            np.testing.assert_allclose(
                wrapped_low_observation[key],
                wrapped_env.observation_space[key].low)

        high_action = type(env.observation_space.spaces)(
            (key, observation_space.high)
            for key, observation_space in env.observation_space.spaces.items())

        high_observation = env.step(high_action)[0]
        wrapped_high_observation = wrapped_env.step(high_action)[0]

        assert (set(env.observation_space.spaces.keys())
                == set(high_observation.keys()))
        assert (set(wrapped_env.observation_space.spaces.keys())
                == set(high_observation.keys()))
        for key in env.observation_space.spaces.keys():
            np.testing.assert_allclose(
                high_observation[key], env.observation_space[key].high)
            np.testing.assert_allclose(
                wrapped_high_observation[key],
                wrapped_env.observation_space[key].high)

    else:
        raise ValueError


@pytest.mark.parametrize("observation_space", [
    UNSCALED_BOX_SPACE,
    spaces.Tuple((UNSCALED_BOX_SPACE, UNSCALED_BOX_SPACE)),
    spaces.Dict({'box-1': UNSCALED_BOX_SPACE, 'box-2': UNSCALED_BOX_SPACE}),
])
def test_raises_on_non_finite_low(observation_space):
    env = FakeEnvironment(observation_space)
    assert isinstance(
        env.observation_space, (spaces.Box, spaces.Tuple, spaces.Dict))

    with pytest.raises(ValueError):
        RescaleObservation(env, -float('inf'), 1.0)

    with pytest.raises(ValueError):
        RescaleObservation(env, -1.0, float('inf'))

    with pytest.raises(ValueError):
        RescaleObservation(env, -1.0, np.nan)


@pytest.mark.parametrize("observation_space", [
    UNSCALED_BOX_SPACE,
    spaces.Tuple((UNSCALED_BOX_SPACE, UNSCALED_BOX_SPACE)),
    spaces.Dict({'box-1': UNSCALED_BOX_SPACE, 'box-2': UNSCALED_BOX_SPACE}),
])
def test_raises_on_high_less_than_low(observation_space):
    env = FakeEnvironment(observation_space)
    assert isinstance(
        env.observation_space, (spaces.Box, spaces.Tuple, spaces.Dict))
    with pytest.raises(ValueError):
        RescaleObservation(env, 1.0, 1.0)
    with pytest.raises(ValueError):
        RescaleObservation(env, 1.0, -1.0)


@pytest.mark.parametrize("observation_space", [
    UNSCALED_BOX_SPACE,
    spaces.Tuple((UNSCALED_BOX_SPACE, UNSCALED_BOX_SPACE)),
    spaces.Dict({'box-1': UNSCALED_BOX_SPACE, 'box-2': UNSCALED_BOX_SPACE}),
])
def test_raises_on_high_equals_low(observation_space):
    env = FakeEnvironment(observation_space)
    assert isinstance(
        env.observation_space, (spaces.Box, spaces.Tuple, spaces.Dict))
    with pytest.raises(ValueError):
        RescaleObservation(env, 1.0, 1.0)


@pytest.mark.parametrize("observation_space", [
    spaces.Discrete(10),
    spaces.Tuple((spaces.Discrete(5), spaces.Discrete(10))),
    spaces.Tuple((
        spaces.Discrete(5),
        spaces.Box(low=np.array((0.0, 0.0)), high=np.array((1.0, 1.0))))),
    spaces.Dict({
        'discrete-5': spaces.Discrete(5),
        'discrete-10': spaces.Discrete(10),
    }),
    spaces.Dict({
        'discrete': spaces.Discrete(5),
        'box': spaces.Box(low=np.array((0.0, 0.0)), high=np.array((1.0, 1.0))),
    }),
])
def test_raises_on_non_box_space(observation_space):
    env = FakeEnvironment(observation_space)
    with pytest.raises(TypeError):
        RescaleObservation(env, -1.0, 1.0)


@pytest.mark.parametrize("observation_space", [
    spaces.Box(low=np.array((0.0, 0.0)), high=np.array((1.0, float('inf')))),
    spaces.Box(low=np.array((0.0, -float('inf'))), high=np.array((1.0, 1.0))),
    spaces.Tuple((
        spaces.Box(low=np.array((0.0, -1.0)), high=np.array((1.0, 1.0))),
        spaces.Box(low=np.array((0.0, -1.0)), high=np.array((1.0, float('inf')))),
    )),
    spaces.Dict({
        'box-1': spaces.Box(low=np.array((0.0, -1.0)), high=np.array((1.0, 1.0))),
        'box-2': spaces.Box(low=np.array((0.0, -float('inf'))), high=np.array((1.0, 1.0))),
    }),
])
def test_raises_on_non_finite_space(observation_space):
    env = FakeEnvironment(observation_space)
    with pytest.raises(ValueError):
        RescaleObservation(env, -1.0, 1.0)
