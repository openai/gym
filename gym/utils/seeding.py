import numpy as np
from numpy.random import Generator
from pkg_resources import parse_version

import gym
from gym import error
from gym.logger import warn


def np_random(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error(
            "Seed must be a non-negative integer or omitted, not {}".format(seed)
        )

    seed_seq = np.random.SeedSequence(seed)
    seed = seed_seq.entropy
    rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
    return rng, seed


# TODO: Remove this class and make it alias to `Generator` in Gym release 0.25.0
# RandomNumberGenerator = np.random.Generator
class RandomNumberGenerator(np.random.Generator):

    # assert parse_version(gym.__version__) < parse_version(REMOVE_SINCE)

    def rand(self, *size):
        warn(
            "Function `rng.rand(*size)` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `Generator.random(size)` instead."
        )

        return self.random(size)

    random_sample = rand

    def randn(self, *size):
        warn(
            "Function `rng.randn(*size)` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.standard_normal(size)` instead."
        )

        return self.standard_normal(size)

    def randint(self, low, high=None, size=None, dtype=int):
        warn(
            "Function `rng.randint(low, [high, size, dtype])` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.integers(low, [high, size, dtype])` instead."
        )

        return self.integers(low=low, high=high, size=size, dtype=dtype)

    random_integers = randint

    def get_state(self):
        warn(
            "Function `rng.get_state()` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.bit_generator.state` instead."
        )

        return self.bit_generator.state

    def set_state(self, state):
        warn(
            "Function `rng.set_state(state)` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.bit_generator.state = state` instead."
        )

        self.bit_generator.state = state

    def seed(self, seed=None):
        warn(
            "Function `rng.seed(seed)` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng = get_rng(seed)` to create a separated generator instead."
        )

        self.bit_generator.state = type(self.bit_generator)(seed).state

    rand.__doc__ = np.random.rand.__doc__
    randn.__doc__ = np.random.randn.__doc__
    randint.__doc__ = np.random.randint.__doc__
    get_state.__doc__ = np.random.get_state.__doc__
    set_state.__doc__ = np.random.set_state.__doc__
    seed.__doc__ = np.random.seed.__doc__


RNG = RandomNumberGenerator
