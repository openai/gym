"""Tests for the pixel observation wrapper."""
from typing import Optional

import numpy as np
import pytest

import gym
from gym import spaces
from gym.wrappers.pixel_observation import STATE_KEY, PixelObservationWrapper


class FakeEnvironment(gym.Env):
    def __init__(self, render_mode="single_rgb_array"):
        self.action_space = spaces.Box(shape=(1,), low=-1, high=1, dtype=np.float32)
        self.render_mode = render_mode

    def render(self, mode="human", width=32, height=32):
        image_shape = (height, width, 3)
        return np.zeros(image_shape, dtype=np.uint8)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        observation = self.observation_space.sample()
        return observation, {}

    def step(self, action):
        del action
        observation = self.observation_space.sample()
        reward, terminal, info = 0.0, False, {}
        return observation, reward, terminal, info


class FakeArrayObservationEnvironment(FakeEnvironment):
    def __init__(self, *args, **kwargs):
        self.observation_space = spaces.Box(
            shape=(2,), low=-1, high=1, dtype=np.float32
        )
        super().__init__(*args, **kwargs)


class FakeDictObservationEnvironment(FakeEnvironment):
    def __init__(self, *args, **kwargs):
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32),
            }
        )
        super().__init__(*args, **kwargs)


@pytest.mark.parametrize("pixels_only", (True, False))
def test_dict_observation(pixels_only):
    pixel_key = "rgb"

    env = FakeDictObservationEnvironment()

    # Make sure we are testing the right environment for the test.
    observation_space = env.observation_space
    assert isinstance(observation_space, spaces.Dict)

    width, height = (320, 240)

    # The wrapper should only add one observation.
    wrapped_env = PixelObservationWrapper(
        env,
        pixel_keys=(pixel_key,),
        pixels_only=pixels_only,
        render_kwargs={pixel_key: {"width": width, "height": height}},
    )

    assert isinstance(wrapped_env.observation_space, spaces.Dict)

    if pixels_only:
        assert len(wrapped_env.observation_space.spaces) == 1
        assert list(wrapped_env.observation_space.spaces.keys()) == [pixel_key]
    else:
        assert (
            len(wrapped_env.observation_space.spaces)
            == len(observation_space.spaces) + 1
        )
        expected_keys = list(observation_space.spaces.keys()) + [pixel_key]
        assert list(wrapped_env.observation_space.spaces.keys()) == expected_keys

    # Check that the added space item is consistent with the added observation.
    observation, info = wrapped_env.reset()
    rgb_observation = observation[pixel_key]

    assert isinstance(info, dict)
    assert rgb_observation.shape == (height, width, 3)
    assert rgb_observation.dtype == np.uint8


@pytest.mark.parametrize("pixels_only", (True, False))
def test_single_array_observation(pixels_only):
    pixel_key = "depth"

    env = FakeArrayObservationEnvironment()
    observation_space = env.observation_space
    assert isinstance(observation_space, spaces.Box)

    wrapped_env = PixelObservationWrapper(
        env, pixel_keys=(pixel_key,), pixels_only=pixels_only
    )
    wrapped_env.observation_space = wrapped_env.observation_space
    assert isinstance(wrapped_env.observation_space, spaces.Dict)

    if pixels_only:
        assert len(wrapped_env.observation_space.spaces) == 1
        assert list(wrapped_env.observation_space.spaces.keys()) == [pixel_key]
    else:
        assert len(wrapped_env.observation_space.spaces) == 2
        assert list(wrapped_env.observation_space.spaces.keys()) == [
            STATE_KEY,
            pixel_key,
        ]

    observation, info = wrapped_env.reset()
    depth_observation = observation[pixel_key]

    assert isinstance(info, dict)
    assert depth_observation.shape == (32, 32, 3)
    assert depth_observation.dtype == np.uint8

    if not pixels_only:
        assert isinstance(observation[STATE_KEY], np.ndarray)
