from collections import OrderedDict
from functools import singledispatch
from typing import Callable, Sequence, Union

import numpy as np

from gym import spaces
from gym.spaces import Space, Box, Discrete, MultiBinary, MultiDiscrete
from gym.vector.utils.spaces import _BaseGymSpaces

__all__ = ["concatenate", "create_empty_array"]


@singledispatch
def concatenate(
    space: Space, items: Sequence, out: Union[tuple, dict, np.ndarray]
) -> Union[tuple, dict, np.ndarray]:
    """Concatenate multiple samples from space into a single object.

    Parameters
    ----------
    space : `gym.spaces.Space` instance
        Observation space of a single environment in the vectorized environment.

    items : iterable of samples of `space`
        Samples to be concatenated.

    out : tuple, dict, or `np.ndarray`
        The output object. This object is a (possibly nested) numpy array.

    Returns
    -------
    out : tuple, dict, or `np.ndarray`
        The output object. This object is a (possibly nested) numpy array.

    Example
    -------
    >>> from gym.spaces import Box
    >>> space = Box(low=0, high=1, shape=(3,), dtype=np.float32)
    >>> out = np.zeros((2, 3), dtype=np.float32)
    >>> items = [space.sample() for _ in range(2)]
    >>> concatenate(space, items, out)
    array([[0.6348213 , 0.28607962, 0.60760117],
           [0.87383074, 0.192658  , 0.2148103 ]], dtype=float32)
    """
    if isinstance(space, (list, tuple)) and isinstance(out, Space):
        # Re-order the arguments, as this was called with positional arguments.
        # This adds backward-compatibility with the previous ordering of args.
        space, items, out = out, space, items  # type: ignore
        return concatenate(space, items, out)

    assert isinstance(items, (list, tuple)), items
    raise ValueError(
        f"Space of type `{0}` is not a valid `gym.Space` instance.".format(type(space))
    )


@concatenate.register(spaces.Box)
@concatenate.register(spaces.Discrete)
@concatenate.register(spaces.MultiDiscrete)
@concatenate.register(spaces.MultiBinary)
def concatenate_base(
    space: Space, items: Union[list, tuple], out: Union[tuple, dict, np.ndarray]
) -> np.ndarray:
    return np.stack(items, axis=0, out=out)


@concatenate.register(spaces.Tuple)
def concatenate_tuple(
    space: spaces.Tuple, items: Union[list, tuple], out: Union[tuple, dict, np.ndarray]
) -> tuple:
    return tuple(
        concatenate(subspace, [item[i] for item in items], out=out[i])
        for (i, subspace) in enumerate(space.spaces)
    )


@concatenate.register(spaces.Dict)
def concatenate_dict(
    space: spaces.Dict, items: Union[list, tuple], out: Union[tuple, dict, np.ndarray]
) -> OrderedDict:
    return OrderedDict(
        [
            (key, concatenate(subspace, [item[key] for item in items], out=out[key]))
            for (key, subspace) in space.spaces.items()
        ]
    )


@concatenate.register(spaces.Space)
def concatenate_custom(
    space: Space, items: Union[list, tuple], out: Union[tuple, dict, np.ndarray]
) -> Union[tuple, dict, np.ndarray]:
    return tuple(items)


@singledispatch
def create_empty_array(
    space: Space, n: int = 1, fn: Callable = np.zeros
) -> Union[tuple, dict, np.ndarray]:
    """Create an empty (possibly nested) numpy array.

    Parameters
    ----------
    space : `gym.spaces.Space` instance
        Observation space of a single environment in the vectorized environment.

    n : int
        Number of environments in the vectorized environment. If `None`, creates
        an empty sample from `space`.

    fn : callable
        Function to apply when creating the empty numpy array. Examples of such
        functions are `np.empty` or `np.zeros`.

    Returns
    -------
    out : tuple, dict, or `np.ndarray`
        The output object. This object is a (possibly nested) numpy array.

    Example
    -------
    >>> from gym.spaces import Box, Dict
    >>> space = Dict({
    ... 'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
    ... 'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)})
    >>> create_empty_array(space, n=2, fn=np.zeros)
    OrderedDict([('position', array([[0., 0., 0.],
                                     [0., 0., 0.]], dtype=float32)),
                 ('velocity', array([[0., 0.],
                                     [0., 0.]], dtype=float32))])
    """
    raise ValueError(
        "Space of type `{0}` is not a valid `gym.Space` "
        "instance.".format(type(space))
    )


@create_empty_array.register(spaces.Box)
@create_empty_array.register(spaces.Discrete)
@create_empty_array.register(spaces.MultiDiscrete)
@create_empty_array.register(spaces.MultiBinary)
def create_empty_array_base(
    space: Space, n: int = 1, fn: Callable = np.zeros
) -> Union[tuple, dict, np.ndarray]:
    shape = space.shape if (n is None) else (n,) + space.shape
    return fn(shape, dtype=space.dtype)


@create_empty_array.register(spaces.Tuple)
def create_empty_array_tuple(
    space: spaces.Tuple, n: int = 1, fn: Callable = np.zeros
) -> tuple:
    return tuple(create_empty_array(subspace, n=n, fn=fn) for subspace in space.spaces)


@create_empty_array.register(spaces.Dict)
def create_empty_array_dict(
    space: spaces.Dict, n: int = 1, fn: Callable = np.zeros
) -> OrderedDict:
    return OrderedDict(
        [
            (key, create_empty_array(subspace, n=n, fn=fn))
            for (key, subspace) in space.spaces.items()
        ]
    )


@create_empty_array.register(Space)
def create_empty_array_custom(
    space: Space, n: int = 1, fn: Callable = np.zeros
) -> tuple:
    return ()
