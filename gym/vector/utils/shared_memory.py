import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union, Iterable

import numpy as np

from gym import logger, spaces
from gym.error import CustomSpaceError
from gym.spaces import Space
from gym.vector.utils.spaces import _BaseGymSpaces

__all__ = ["create_shared_memory", "read_from_shared_memory", "write_to_shared_memory"]


@singledispatch
def create_shared_memory(space: Space, n: int = 1, ctx=mp):
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
        f"Cannot create a shared memory for space with type `{type(space)}`. "
        f"Shared memory only supports default Gym spaces (e.g. `Box`, `Tuple`, "
        f"`Dict`, etc...) out-of-the-box. To use custom spaces, register a "
        f"function to use for spaces of type {type(space)} by decorating it "
        f" with `create_shared_memory.register({type(space).__name__})`."
    )


@create_shared_memory.register(spaces.Box)
@create_shared_memory.register(spaces.Discrete)
@create_shared_memory.register(spaces.MultiDiscrete)
@create_shared_memory.register(spaces.MultiBinary)
def _create_base_shared_memory(space: Space, n: int = 1, ctx=mp) -> mp.Array:
    dtype = space.dtype.char
    if dtype in "?":
        dtype = c_bool
    return ctx.Array(dtype, n * int(np.prod(space.shape)))


@create_shared_memory.register(spaces.Tuple)
def _create_tuple_shared_memory(space: spaces.Tuple, n: int = 1, ctx=mp) -> tuple:
    return tuple(create_shared_memory(subspace, n=n, ctx=ctx)
        for subspace in space.spaces)


@create_shared_memory.register(spaces.Dict)
def _create_dict_shared_memory(space: spaces.Dict, n: int = 1, ctx=mp) -> OrderedDict:
    return OrderedDict([(key, create_shared_memory(subspace, n=n, ctx=ctx))
        for (key, subspace) in space.spaces.items()])


@singledispatch
def read_from_shared_memory(space: Space,
                            shared_memory: Union[dict, tuple, mp.Array],
                            n: int = 1) -> Union[dict, tuple, mp.Array]:
    """Read the batch of observations from shared memory as a numpy array.

    Parameters
    ----------
    space : `gym.spaces.Space` instance
        Observation space of a single environment in the vectorized environment.

    shared_memory : dict, tuple, or `multiprocessing.Array` instance
        Shared object across processes. This contains the observations from the
        vectorized environment. This object is created with `create_shared_memory`.

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
    if isinstance(shared_memory, Space):
        # Re-order the positional arguments to keep this backward-compatible.
        shared_memory, space, n = space, shared_memory, n  # type: ignore
        return read_from_shared_memory(space, shared_memory, n)

    raise CustomSpaceError(
        f"Cannot read from a shared memory for space with type "
        f"`{type(space)}`. Shared memory only supports default Gym spaces "
        f"(e.g. `Box`, `Tuple`, `Dict`, etc...) out-of-the-box. To use custom "
        f"spaces, register a function to use for spaces of type {type(space)} "
        f"by decorating it with "
        f"`read_from_shared_memory.register({type(space).__name__})`."
    )


@read_from_shared_memory.register(spaces.Box)
@read_from_shared_memory.register(spaces.Discrete)
@read_from_shared_memory.register(spaces.MultiDiscrete)
@read_from_shared_memory.register(spaces.MultiBinary)
def _read_base_from_shared_memory(space: Space,
                                 shared_memory: mp.Array,
                                 n: int = 1) -> np.ndarray:
    return np.frombuffer(shared_memory.get_obj(),
        dtype=space.dtype).reshape((n,) + space.shape)


@read_from_shared_memory.register(spaces.Tuple)
def _read_tuple_from_shared_memory(space: spaces.Tuple,
                                  shared_memory: tuple,
                                  n: int = 1) -> tuple:
    return tuple(read_from_shared_memory(memory, subspace, n=n)
        for (memory, subspace) in zip(shared_memory, space.spaces))


@read_from_shared_memory.register(spaces.Dict)
def _read_dict_from_shared_memory(space: spaces.Dict,
                                 shared_memory: dict,
                                 n: int = 1) -> OrderedDict:
    return OrderedDict([(key, read_from_shared_memory(shared_memory[key],
        subspace, n=n)) for (key, subspace) in space.spaces.items()])


@singledispatch
def write_to_shared_memory(space: Space,
                           index: int,
                           value,
                           shared_memory: Union[dict, tuple, mp.Array]) -> None:
    """Write the observation of a single environment into shared memory.

    Parameters
    ----------
    space : `gym.spaces.Space` instance
        Observation space of a single environment in the vectorized environment.

    index : int
        Index of the environment (must be in `[0, num_envs)`).

    value : sample from `space`
        Observation of the single environment to write to shared memory.

    shared_memory : dict, tuple, or `multiprocessing.Array` instance
        Shared object across processes. This contains the observations from the
        vectorized environment. This object is created with `create_shared_memory`.

    Returns
    -------
    `None`
    """
    if isinstance(space, int) and isinstance(shared_memory, Space):
        # Reorder the arguments, to keep this backward compatible.
        index, value, shared_memory, space = space, index, value, shared_memory  # type: ignore
        return write_to_shared_memory(space, index, value, shared_memory)
    
    raise CustomSpaceError(
        f"Cannot write to a shared memory for space with type `{type(space)}`. "
        f"Shared memory only supports default Gym spaces (e.g. `Box`, `Tuple`, "
        f"`Dict`, etc...) out-of-the-box. To use custom spaces, register a "
        f"function to use for spaces of type {type(space)} by decorating it "
        f" with `write_to_shared_memory.register({type(space).__name__})`."
    )


@write_to_shared_memory.register(spaces.Discrete)
@write_to_shared_memory.register(spaces.Box)
@write_to_shared_memory.register(spaces.MultiDiscrete)
@write_to_shared_memory.register(spaces.MultiBinary)
def _write_base_to_shared_memory(space: Space,
                                index: int,
                                value,
                                shared_memory: mp.Array) -> None:
    size = int(np.prod(space.shape))
    destination = np.frombuffer(shared_memory.get_obj(), dtype=space.dtype)
    np.copyto(
        destination[index * size : (index + 1) * size],
        np.asarray(value, dtype=space.dtype).flatten(),
    )



@write_to_shared_memory.register(spaces.Tuple)
def _write_tuple_to_shared_memory(space: spaces.Tuple,
                                 index: int,
                                 values: Iterable,
                                 shared_memory: tuple) -> None:
    for value, memory, subspace in zip(values, shared_memory, space.spaces):
        write_to_shared_memory(index, value, memory, subspace)


@write_to_shared_memory.register(spaces.Dict)
def _write_dict_to_shared_memory(space: spaces.Dict,
                                index: int,
                                values: dict,
                                shared_memory: dict) -> None:
    for key, subspace in space.spaces.items():
        write_to_shared_memory(index, values[key], shared_memory[key], subspace)
