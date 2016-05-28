import numpy as np
import os
import random as _random
import struct

def random(seed=None):
    seed = _random_seed(seed)

    rng = _random.Random()
    rng.seed(seed)
    return rng

def np_random(seed=None):
    seed = _np_random_seed(seed)

    rng = np.random.RandomState()
    rng.seed(seed)
    return rng

def uint_32_seed(seed=None):
    """Create a uint32 random seed.

    Args:
        seed (Optional[int, str]): The base form of the seed to use
    """
    return _seed(seed) % 2 ** 32

def _random_seed(seed):
    return _seed(seed)

def _np_random_seed(seed):
    return _int_list_from_bigint(_seed(seed))

def _seed(a=None):
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.

    Args:
        a (Optional[int, str]): None seeds from an operating system specific randomness source. If an int or str passed, all of the bits are used.
    """
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(32))

    if isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a)

    return a

# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b'\0' * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def _int_list_from_bigint(bigint):
    # Special case 0
    if bigint < 0:
        raise error.Error('Seed must be non-negative, not {}'.format(bigint))
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints
