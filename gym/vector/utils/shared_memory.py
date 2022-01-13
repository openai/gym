import numpy as np
import multiprocessing as mp
from ctypes import c_bool
from collections import OrderedDict

from gym import logger
from gym.spaces import Space, Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict
from gym.error import CustomSpaceError
from gym.vector.utils.spaces import _BaseGymSpaces

from functools import singledispatch

__all__ = ["create_shared_memory", "read_from_shared_memory", "write_to_shared_memory"]


@singledispatch
def create_shared_memory(space, n=1, ctx=mp):
    """Create a shared memory object, to be shared across processes. This
    eventually contains the observations from the vectorized environment.

    Parameters
    ----------
    space : `gym.spaces.Space` instance
        Observation space of a single environment in the vectorized environment.

    n : int
        Number of environments in the vectorized environment (i.e. the number
        of processes).

    ctx : `multiprocessing` context
        Context for multiprocessing.

    Returns
    -------
    shared_memory : dict, tuple, or `multiprocessing.Array` instance
        Shared object across processes.
    """
    raise CustomSpaceError(
        "Cannot create a shared memory for space with "
        "type `{}`. Shared memory only supports "
        "default Gym spaces (e.g. `Box`, `Tuple`, "
        "`Dict`, etc...), and does not support custom "
        "Gym spaces.".format(type(space))
    )


@create_shared_memory.register(Box)
@create_shared_memory.register(Discrete)
@create_shared_memory.register(MultiDiscrete)
@create_shared_memory.register(MultiBinary)
def _create_base_shared_memory(space, n=1, ctx=mp):
    dtype = space.dtype.char
    if dtype in "?":
        dtype = c_bool
    return ctx.Array(dtype, n * int(np.prod(space.shape)))


@create_shared_memory.register(Tuple)
def _create_tuple_shared_memory(space, n=1, ctx=mp):
    return tuple(
        create_shared_memory(subspace, n=n, ctx=ctx) for subspace in space.spaces
    )


@create_shared_memory.register(Dict)
def _create_dict_shared_memory(space, n=1, ctx=mp):
    return OrderedDict(
        [
            (key, create_shared_memory(subspace, n=n, ctx=ctx))
            for (key, subspace) in space.spaces.items()
        ]
    )


@singledispatch
def read_from_shared_memory(space, shared_memory, n=1):
    """Read the batch of observations from shared memory as a numpy array.

    Parameters
    ----------
    shared_memory : dict, tuple, or `multiprocessing.Array` instance
        Shared object across processes. This contains the observations from the
        vectorized environment. This object is created with `create_shared_memory`.

    space : `gym.spaces.Space` instance
        Observation space of a single environment in the vectorized environment.

    n : int
        Number of environments in the vectorized environment (i.e. the number
        of processes).

    Returns
    -------
    observations : dict, tuple or `np.ndarray` instance
        Batch of observations as a (possibly nested) numpy array.

    Notes
    -----
    The numpy array objects returned by `read_from_shared_memory` shares the
    memory of `shared_memory`. Any changes to `shared_memory` are forwarded
    to `observations`, and vice-versa. To avoid any side-effect, use `np.copy`.
    """
    raise CustomSpaceError(
        "Cannot read from a shared memory for space with "
        "type `{}`. Shared memory only supports "
        "default Gym spaces (e.g. `Box`, `Tuple`, "
        "`Dict`, etc...), and does not support custom "
        "Gym spaces.".format(type(space))
    )


@read_from_shared_memory.register(Box)
@read_from_shared_memory.register(Discrete)
@read_from_shared_memory.register(MultiDiscrete)
@read_from_shared_memory.register(MultiBinary)
def _read_base_from_shared_memory(space, shared_memory, n=1):
    return np.frombuffer(shared_memory.get_obj(), dtype=space.dtype).reshape(
        (n,) + space.shape
    )


@read_from_shared_memory.register(Tuple)
def _read_tuple_from_shared_memory(space, shared_memory, n=1):
    return tuple(
        read_from_shared_memory(subspace, memory, n=n)
        for (memory, subspace) in zip(shared_memory, space.spaces)
    )


@read_from_shared_memory.register(Dict)
def _read_dict_from_shared_memory(space, shared_memory, n=1):
    return OrderedDict(
        [
            (key, read_from_shared_memory(subspace, shared_memory[key], n=n))
            for (key, subspace) in space.spaces.items()
        ]
    )


@singledispatch
def write_to_shared_memory(space, index, value, shared_memory):
    """Write the observation of a single environment into shared memory.

    Parameters
    ----------
    index : int
        Index of the environment (must be in `[0, num_envs)`).

    value : sample from `space`
        Observation of the single environment to write to shared memory.

    shared_memory : dict, tuple, or `multiprocessing.Array` instance
        Shared object across processes. This contains the observations from the
        vectorized environment. This object is created with `create_shared_memory`.

    space : `gym.spaces.Space` instance
        Observation space of a single environment in the vectorized environment.

    Returns
    -------
    `None`
    """
    raise CustomSpaceError(
        "Cannot write to a shared memory for space with "
        "type `{}`. Shared memory only supports "
        "default Gym spaces (e.g. `Box`, `Tuple`, "
        "`Dict`, etc...), and does not support custom "
        "Gym spaces.".format(type(space))
    )


@write_to_shared_memory.register(Box)
@write_to_shared_memory.register(Discrete)
@write_to_shared_memory.register(MultiDiscrete)
@write_to_shared_memory.register(MultiBinary)
def _write_base_to_shared_memory(space, index, value, shared_memory):
    size = int(np.prod(space.shape))
    destination = np.frombuffer(shared_memory.get_obj(), dtype=space.dtype)
    np.copyto(
        destination[index * size : (index + 1) * size],
        np.asarray(value, dtype=space.dtype).flatten(),
    )


@write_to_shared_memory.register(Tuple)
def _write_tuple_to_shared_memory(space, index, values, shared_memory):
    for value, memory, subspace in zip(values, shared_memory, space.spaces):
        write_to_shared_memory(subspace, index, value, memory)


@write_to_shared_memory.register(Dict)
def _write_dict_to_shared_memory(space, index, values, shared_memory):
    for key, subspace in space.spaces.items():
        write_to_shared_memory(subspace, index, values[key], shared_memory[key])
