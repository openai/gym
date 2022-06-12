"""Utility functions for vector environments to share memory between processes."""
import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union

import numpy as np

from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple

__all__ = ["create_shared_memory", "read_from_shared_memory", "write_to_shared_memory"]


@singledispatch
def create_shared_memory(
    space: Space, n: int = 1, ctx=mp
) -> Union[dict, tuple, mp.Array]:
    """Create a shared memory object, to be shared across processes.

    This eventually contains the observations from the vectorized environment.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment (i.e. the number of processes).
        ctx: The multiprocess module

    Returns:
        shared_memory for the shared object across processes.

    Raises:
        CustomSpaceError: Space is not a valid :class:`gym.Space` instance
    """
    raise CustomSpaceError(
        "Cannot create a shared memory for space with "
        f"type `{type(space)}`. Shared memory only supports "
        "default Gym spaces (e.g. `Box`, `Tuple`, "
        "`Dict`, etc...), and does not support custom "
        "Gym spaces."
    )


@create_shared_memory.register(Box)
@create_shared_memory.register(Discrete)
@create_shared_memory.register(MultiDiscrete)
@create_shared_memory.register(MultiBinary)
def _create_base_shared_memory(space, n: int = 1, ctx=mp):
    dtype = space.dtype.char
    if dtype in "?":
        dtype = c_bool
    return ctx.Array(dtype, n * int(np.prod(space.shape)))


@create_shared_memory.register(Tuple)
def _create_tuple_shared_memory(space, n: int = 1, ctx=mp):
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
def read_from_shared_memory(
    space: Space, shared_memory: Union[dict, tuple, mp.Array], n: int = 1
) -> Union[dict, tuple, np.ndarray]:
    """Read the batch of observations from shared memory as a numpy array.

    ..notes::
        The numpy array objects returned by `read_from_shared_memory` shares the
        memory of `shared_memory`. Any changes to `shared_memory` are forwarded
        to `observations`, and vice-versa. To avoid any side-effect, use `np.copy`.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        shared_memory: Shared object across processes. This contains the observations from the vectorized environment.
            This object is created with `create_shared_memory`.
        n: Number of environments in the vectorized environment (i.e. the number of processes).

    Returns:
        Batch of observations as a (possibly nested) numpy array.

    Raises:
        CustomSpaceError: Space is not a valid :class:`gym.Space` instance
    """
    raise CustomSpaceError(
        "Cannot read from a shared memory for space with "
        f"type `{type(space)}`. Shared memory only supports "
        "default Gym spaces (e.g. `Box`, `Tuple`, "
        "`Dict`, etc...), and does not support custom "
        "Gym spaces."
    )


@read_from_shared_memory.register(Box)
@read_from_shared_memory.register(Discrete)
@read_from_shared_memory.register(MultiDiscrete)
@read_from_shared_memory.register(MultiBinary)
def _read_base_from_shared_memory(space, shared_memory, n: int = 1):
    return np.frombuffer(shared_memory.get_obj(), dtype=space.dtype).reshape(
        (n,) + space.shape
    )


@read_from_shared_memory.register(Tuple)
def _read_tuple_from_shared_memory(space, shared_memory, n: int = 1):
    return tuple(
        read_from_shared_memory(subspace, memory, n=n)
        for (memory, subspace) in zip(shared_memory, space.spaces)
    )


@read_from_shared_memory.register(Dict)
def _read_dict_from_shared_memory(space, shared_memory, n: int = 1):
    return OrderedDict(
        [
            (key, read_from_shared_memory(subspace, shared_memory[key], n=n))
            for (key, subspace) in space.spaces.items()
        ]
    )


@singledispatch
def write_to_shared_memory(
    space: Space,
    index: int,
    value: np.ndarray,
    shared_memory: Union[dict, tuple, mp.Array],
):
    """Write the observation of a single environment into shared memory.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        index: Index of the environment (must be in `[0, num_envs)`).
        value: Observation of the single environment to write to shared memory.
        shared_memory: Shared object across processes. This contains the observations from the vectorized environment.
            This object is created with `create_shared_memory`.

    Raises:
        CustomSpaceError: Space is not a valid :class:`gym.Space` instance
    """
    raise CustomSpaceError(
        "Cannot write to a shared memory for space with "
        f"type `{type(space)}`. Shared memory only supports "
        "default Gym spaces (e.g. `Box`, `Tuple`, "
        "`Dict`, etc...), and does not support custom "
        "Gym spaces."
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
