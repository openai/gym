import time
from typing import Optional

import numpy as np

import gym
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple
from gym.utils.seeding import RandomNumberGenerator

spaces = [
    Box(low=np.array(-1.0), high=np.array(1.0), dtype=np.float64),
    Box(low=np.array([0.0]), high=np.array([10.0]), dtype=np.float64),
    Box(
        low=np.array([-1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64
    ),
    Box(
        low=np.array([[-1.0, 0.0], [0.0, -1.0]]), high=np.ones((2, 2)), dtype=np.float64
    ),
    Box(low=0, high=255, shape=(), dtype=np.uint8),
    Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
    Discrete(2),
    Discrete(5, start=-2),
    Tuple((Discrete(3), Discrete(5))),
    Tuple(
        (
            Discrete(7),
            Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64),
        )
    ),
    MultiDiscrete([11, 13, 17]),
    MultiBinary(19),
    Dict(
        {
            "position": Discrete(23),
            "velocity": Box(
                low=np.array([0.0]), high=np.array([1.0]), dtype=np.float64
            ),
        }
    ),
    Dict(
        {
            "position": Dict({"x": Discrete(29), "y": Discrete(31)}),
            "velocity": Tuple(
                (Discrete(37), Box(low=0, high=255, shape=(), dtype=np.uint8))
            ),
        }
    ),
]

HEIGHT, WIDTH = 64, 64


class UnittestSlowEnv(gym.Env):
    def __init__(self, slow_reset=0.3):
        super().__init__()
        self.slow_reset = slow_reset
        self.observation_space = Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8
        )
        self.action_space = Box(low=0.0, high=1.0, shape=(), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.slow_reset > 0:
            time.sleep(self.slow_reset)
        return self.observation_space.sample()

    def step(self, action):
        time.sleep(action)
        observation = self.observation_space.sample()
        reward, done = 0.0, False
        return observation, reward, done, {}


class CustomSpace(gym.Space):
    """Minimal custom observation space."""

    def sample(self):
        return self.np_random.integers(0, 10, ())

    def contains(self, x):
        return 0 <= x <= 10

    def __eq__(self, other):
        return isinstance(other, CustomSpace)


custom_spaces = [
    CustomSpace(),
    Tuple((CustomSpace(), Box(low=0, high=255, shape=(), dtype=np.uint8))),
]


class CustomSpaceEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = CustomSpace()
        self.action_space = CustomSpace()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        return "reset"

    def step(self, action):
        observation = f"step({action:s})"
        reward, done = 0.0, False
        return observation, reward, done, {}


def make_env(env_name, seed, **kwargs):
    def _make():
        env = gym.make(env_name, disable_env_checker=True, **kwargs)
        env.action_space.seed(seed)
        env.reset(seed=seed)
        return env

    return _make


def make_slow_env(slow_reset, seed):
    def _make():
        env = UnittestSlowEnv(slow_reset=slow_reset)
        env.reset(seed=seed)
        return env

    return _make


def make_custom_space_env(seed):
    def _make():
        env = CustomSpaceEnv()
        env.reset(seed=seed)
        return env

    return _make


def assert_rng_equal(rng_1: RandomNumberGenerator, rng_2: RandomNumberGenerator):
    assert rng_1.bit_generator.state == rng_2.bit_generator.state
