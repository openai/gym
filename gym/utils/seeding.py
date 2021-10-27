import hashlib
import os
import struct
from typing import Optional

import numpy as np
from numpy.random import Generator

from gym import error
from gym.logger import warn


def np_random(seed: Optional[int] = None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error(
            "Seed must be a non-negative integer or omitted, not {}".format(seed)
        )

    seed_seq = np.random.SeedSequence(seed)
    seed = seed_seq.entropy
    rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
    return rng, seed


# TODO: Remove this class and make it alias to `Generator` in a future Gym release
# RandomNumberGenerator = np.random.Generator
class RandomNumberGenerator(np.random.Generator):
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
            "Please use `rng, seed = gym.utils.seeding.np_random(seed)` to create a separate generator instead."
        )

        self.bit_generator.state = type(self.bit_generator)(seed).state

    rand.__doc__ = np.random.rand.__doc__
    randn.__doc__ = np.random.randn.__doc__
    randint.__doc__ = np.random.randint.__doc__
    get_state.__doc__ = np.random.get_state.__doc__
    set_state.__doc__ = np.random.set_state.__doc__
    seed.__doc__ = np.random.seed.__doc__


RNG = RandomNumberGenerator

# Legacy functions


def hash_seed(seed=None, max_bytes=8):
    """Any given evaluation is likely to have many PRNG's active at
    once. (Most commonly, because the environment is running in
    multiple processes.) There's literature indicating that having
    linear correlations between seeds of multiple PRNG's can correlate
    the outputs:
    http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
    http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
    http://dl.acm.org/citation.cfm?id=1276928
    Thus, for sanity we hash the seeds before using them. (This scheme
    is likely not crypto-strength, but it should be good enough to get
    rid of simple correlations.)
    Args:
        seed (Optional[int]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    warn(
        "Function `hash_seed(seed, max_bytes)` is marked as deprecated and will be removed in the future. "
    )
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode("utf8")).digest()
    return _bigint_from_bytes(hash[:max_bytes])


def create_seed(a=None, max_bytes=8):
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.
    Args:
        a (Optional[int, str]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    warn(
        "Function `create_seed(a, max_bytes)` is marked as deprecated and will be removed in the future. "
    )
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode("utf8")
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, int):
        a = a % 2 ** (8 * max_bytes)
    else:
        raise error.Error("Invalid type for seed: {} ({})".format(type(a), a))

    return a


# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bytes):
    warn(
        "Function `_bigint_from_bytes(bytes)` is marked as deprecated and will be removed in the future. "
    )
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b"\0" * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum


def _int_list_from_bigint(bigint):
    warn(
        "Function `_int_list_from_bigint` is marked as deprecated and will be removed in the future. "
    )
    # Special case 0
    if bigint < 0:
        raise error.Error("Seed must be non-negative, not {}".format(bigint))
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints
