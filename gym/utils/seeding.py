"""Set of random number generator functions: seeding, generator, hashing seeds."""
import hashlib
import os
import struct
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from gym import error
from gym.logger import deprecation


def np_random(seed: Optional[int] = None) -> Tuple["RandomNumberGenerator", Any]:
    """Generates a random number generator from the seed and returns the Generator and seed.

    Args:
        seed: The seed used to create the generator

    Returns:
        The generator and resulting seed

    Raises:
        Error: Seed must be a non-negative integer or omitted
    """
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error(f"Seed must be a non-negative integer or omitted, not {seed}")

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
    return rng, np_seed


# TODO: Remove this class and make it alias to `Generator` in a future Gym release
# RandomNumberGenerator = np.random.Generator
class RandomNumberGenerator(np.random.Generator):
    """Random number generator class that inherits from numpy's random Generator class."""

    def rand(self, *size):
        """Deprecated rand function using random."""
        deprecation(
            "Function `rng.rand(*size)` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `Generator.random(size)` instead."
        )

        return self.random(size)

    random_sample = rand

    def randn(self, *size):
        """Deprecated random standard normal function use standard_normal."""
        deprecation(
            "Function `rng.randn(*size)` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.standard_normal(size)` instead."
        )

        return self.standard_normal(size)

    def randint(self, low, high=None, size=None, dtype=int):
        """Deprecated random integer function use integers."""
        deprecation(
            "Function `rng.randint(low, [high, size, dtype])` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.integers(low, [high, size, dtype])` instead."
        )

        return self.integers(low=low, high=high, size=size, dtype=dtype)

    random_integers = randint

    def get_state(self):
        """Deprecated get rng state use bit_generator.state."""
        deprecation(
            "Function `rng.get_state()` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.bit_generator.state` instead."
        )

        return self.bit_generator.state

    def set_state(self, state):
        """Deprecated set rng state function use bit_generator.state = state."""
        deprecation(
            "Function `rng.set_state(state)` is marked as deprecated "
            "and will be removed in the future. "
            "Please use `rng.bit_generator.state = state` instead."
        )

        self.bit_generator.state = state

    def seed(self, seed=None):
        """Deprecated seed function use gym.utils.seeding.np_random(seed)."""
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

    def __reduce__(self):
        """Reduces the Random Number Generator to a RandomNumberGenerator, init_args and additional args."""
        # np.random.Generator defines __reduce__, but it's hard-coded to
        # return a Generator instead of its subclass RandomNumberGenerator.
        # We need to override it here, otherwise sampling from a Space will
        # be broken after pickling and unpickling, due to using the deprecated
        # methods defined above.
        # See: https://github.com/numpy/numpy/blob/41d37b714caa1eef72f984d529f1d40ed48ce535/numpy/random/_generator.pyx#L221-L223
        # And: https://github.com/numpy/numpy/blob/41d37b714caa1eef72f984d529f1d40ed48ce535/numpy/random/_pickle.py#L17-L37
        _, init_args, *args = np.random.Generator.__reduce__(self)
        return (RandomNumberGenerator._generator_ctor, init_args, *args)

    @staticmethod
    def _generator_ctor(bit_generator_name="MT19937"):
        # Workaround method for RandomNumberGenerator pickling, see __reduce__ above.
        # Ported from numpy.random._pickle.__generator_ctor function.
        from numpy.random._pickle import BitGenerators

        if bit_generator_name in BitGenerators:
            bit_generator = BitGenerators[bit_generator_name]
        else:
            raise ValueError(
                f"{bit_generator_name} is not a known BitGenerator module."
            )
        return RandomNumberGenerator(bit_generator())


RNG = RandomNumberGenerator

# Legacy functions


def hash_seed(seed: Optional[int] = None, max_bytes: int = 8) -> int:
    """Any given evaluation is likely to have many PRNG's active at once.

    (Most commonly, because the environment is running in multiple processes.)
    There's literature indicating that having linear correlations between seeds of multiple PRNG's can correlate the outputs:
        http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
        http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
        http://dl.acm.org/citation.cfm?id=1276928
    Thus, for sanity we hash the seeds before using them. (This scheme is likely not crypto-strength, but it should be good enough to get rid of simple correlations.)

    Args:
        seed: None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.

    Returns:
        The hashed seed
    """
    deprecation(
        "Function `hash_seed(seed, max_bytes)` is marked as deprecated and will be removed in the future. "
    )
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode("utf8")).digest()
    return _bigint_from_bytes(hash[:max_bytes])


def create_seed(a: Optional[Union[int, str]] = None, max_bytes: int = 8) -> int:
    """Create a strong random seed.

    Otherwise, Python 2 would seed using the system time, which might be non-robust especially in the presence of concurrency.

    Args:
        a: None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.

    Returns:
        A seed

    Raises:
        Error: Invalid type for seed, expects None or str or int
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
        bigint, mod = divmod(bigint, 2**32)
        ints.append(mod)
    return ints
