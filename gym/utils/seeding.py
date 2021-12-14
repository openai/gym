import hashlib
from typing import Optional, List, Union
import os
import struct

import numpy as np
from numpy.random import Generator

from gym import error
from gym.logger import deprecation


def np_random(seed: Optional[int] = None) -> tuple:
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error(f"Seed must be a non-negative integer or omitted, not {seed}")

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
    return rng, np_seed


# TODO: Remove this class and make it alias to `Generator` in a future Gym release
# RandomNumberGenerator = np.random.Generator
class RandomNumberGenerator(np.random.Generator):
    def rand(self, *size):
        deprecation(
            "Function `rng.rand(*size)` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `Generator.random(size)` instead."
        )

        return self.random(size)

    random_sample = rand

    def randn(self, *size):
        deprecation(
            "Function `rng.randn(*size)` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.standard_normal(size)` instead."
        )

        return self.standard_normal(size)

    def randint(self, low, high=None, size=None, dtype=int):
        deprecation(
            "Function `rng.randint(low, [high, size, dtype])` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.integers(low, [high, size, dtype])` instead."
        )

        return self.integers(low=low, high=high, size=size, dtype=dtype)

    random_integers = randint

    def get_state(self):
        deprecation(
            "Function `rng.get_state()` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.bit_generator.state` instead."
        )

        return self.bit_generator.state

    def set_state(self, state):
        deprecation(
            "Function `rng.set_state(state)` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.bit_generator.state = state` instead."
        )

        self.bit_generator.state = state

    def seed(self, seed=None):
        deprecation(
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


def hash_seed(seed: Optional[int] = None, max_bytes: int = 8) -> int:
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
        seed: None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    deprecation(
        "Function `hash_seed(seed, max_bytes)` is marked as deprecated and will be removed in the future. "
    )
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode("utf8")).digest()
    return _bigint_from_bytes(hash[:max_bytes])


def create_seed(a: Optional[Union[int, str]] = None, max_bytes: int = 8) -> int:
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.
    Args:
        a: None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    deprecation(
        "Function `create_seed(a, max_bytes)` is marked as deprecated and will be removed in the future. "
    )
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        bt = a.encode("utf8")
        bt += hashlib.sha512(bt).digest()
        a = _bigint_from_bytes(bt[:max_bytes])
    elif isinstance(a, int):
        a = int(a % 2 ** (8 * max_bytes))
    else:
        raise error.Error(f"Invalid type for seed: {type(a)} ({a})")

    return a


# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bt: bytes) -> int:
    deprecation(
        "Function `_bigint_from_bytes(bytes)` is marked as deprecated and will be removed in the future. "
    )
    sizeof_int = 4
    padding = sizeof_int - len(bt) % sizeof_int
    bt += b"\0" * padding
    int_count = int(len(bt) / sizeof_int)
    unpacked = struct.unpack(f"{int_count}I", bt)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum


def _int_list_from_bigint(bigint: int) -> List[int]:
    deprecation(
        "Function `_int_list_from_bigint` is marked as deprecated and will be removed in the future. "
    )
    # Special case 0
    if bigint < 0:
        raise error.Error(f"Seed must be non-negative, not {bigint}")
    elif bigint == 0:
        return [0]

    ints: List[int] = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints
