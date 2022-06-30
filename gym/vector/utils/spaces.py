"""Utility functions for gym spaces: batch space and iterator."""
from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator

import numpy as np

from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple

BaseGymSpaces = (Box, Discrete, MultiDiscrete, MultiBinary)
_BaseGymSpaces = BaseGymSpaces
__all__ = ["BaseGymSpaces", "_BaseGymSpaces", "batch_space", "iterate"]


@singledispatch
def batch_space(space: Space, n: int = 1) -> Space:
    """Create a (batched) space, containing multiple copies of a single space.

    Example::

        >>> from gym.spaces import Box, Dict
        >>> space = Dict({
        ...     'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
        ...     'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)
        ... })
        >>> batch_space(space, n=5)
        Dict(position:Box(5, 3), velocity:Box(5, 2))

    Args:
        space: Space (e.g. the observation space) for a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment.

    Returns:
        Space (e.g. the observation space) for a batch of environments in the vectorized environment.

    Raises:
        ValueError: Cannot batch space that is not a valid :class:`gym.Space` instance
    """
    raise ValueError(
        f"Cannot batch space with type `{type(space)}`. The space must be a valid `gym.Space` instance."
    )


@batch_space.register(Box)
def _batch_space_box(space, n=1):
    repeats = tuple([n] + [1] * space.low.ndim)
    low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
    return Box(low=low, high=high, dtype=space.dtype, seed=deepcopy(space.np_random))


@batch_space.register(Discrete)
def _batch_space_discrete(space, n=1):
    if space.start == 0:
        return MultiDiscrete(
            np.full((n,), space.n, dtype=space.dtype),
            dtype=space.dtype,
            seed=deepcopy(space.np_random),
        )
    else:
        return Box(
            low=space.start,
            high=space.start + space.n - 1,
            shape=(n,),
            dtype=space.dtype,
            seed=deepcopy(space.np_random),
        )


@batch_space.register(MultiDiscrete)
def _batch_space_multidiscrete(space, n=1):
    repeats = tuple([n] + [1] * space.nvec.ndim)
    high = np.tile(space.nvec, repeats) - 1
    return Box(
        low=np.zeros_like(high),
        high=high,
        dtype=space.dtype,
        seed=deepcopy(space.np_random),
    )


@batch_space.register(MultiBinary)
def _batch_space_multibinary(space, n=1):
    return Box(
        low=0,
        high=1,
        shape=(n,) + space.shape,
        dtype=space.dtype,
        seed=deepcopy(space.np_random),
    )


@batch_space.register(Tuple)
def _batch_space_tuple(space, n=1):
    return Tuple(
        tuple(batch_space(subspace, n=n) for subspace in space.spaces),
        seed=deepcopy(space.np_random),
    )


@batch_space.register(Dict)
def _batch_space_dict(space, n=1):
    return Dict(
        OrderedDict(
            [
                (key, batch_space(subspace, n=n))
                for (key, subspace) in space.spaces.items()
            ]
        ),
        seed=deepcopy(space.np_random),
    )


@batch_space.register(Space)
def _batch_space_custom(space, n=1):
    # Without deepcopy, then the space.np_random is batched_space.spaces[0].np_random
    # Which is an issue if you are sampling actions of both the original space and the batched space
    batched_space = Tuple(
        tuple(deepcopy(space) for _ in range(n)), seed=deepcopy(space.np_random)
    )
    new_seeds = list(map(int, batched_space.np_random.integers(0, 1e8, n)))
    batched_space.seed(new_seeds)
    return batched_space


@singledispatch
def iterate(space: Space, items) -> Iterator:
    """Iterate over the elements of a (batched) space.

    Example::

        >>> from gym.spaces import Box, Dict
        >>> space = Dict({
        ... 'position': Box(low=0, high=1, shape=(2, 3), dtype=np.float32),
        ... 'velocity': Box(low=0, high=1, shape=(2, 2), dtype=np.float32)})
        >>> items = space.sample()
        >>> it = iterate(space, items)
        >>> next(it)
        {'position': array([-0.99644893, -0.08304597, -0.7238421 ], dtype=float32),
        'velocity': array([0.35848552, 0.1533453 ], dtype=float32)}
        >>> next(it)
        {'position': array([-0.67958736, -0.49076623,  0.38661423], dtype=float32),
        'velocity': array([0.7975036 , 0.93317133], dtype=float32)}
        >>> next(it)
        StopIteration

    Args:
        space: Space to which `items` belong to.
        items: Items to be iterated over.

    Returns:
        Iterator over the elements in `items`.

    Raises:
        ValueError: Space is not an instance of :class:`gym.Space`
    """
    raise ValueError(
        f"Space of type `{type(space)}` is not a valid `gym.Space` instance."
    )


@iterate.register(Discrete)
def _iterate_discrete(space, items):
    raise TypeError("Unable to iterate over a space of type `Discrete`.")


@iterate.register(Box)
@iterate.register(MultiDiscrete)
@iterate.register(MultiBinary)
def _iterate_base(space, items):
    try:
        return iter(items)
    except TypeError:
        raise TypeError(f"Unable to iterate over the following elements: {items}")


@iterate.register(Tuple)
def _iterate_tuple(space, items):
    # If this is a tuple of custom subspaces only, then simply iterate over items
    if all(
        isinstance(subspace, Space)
        and (not isinstance(subspace, BaseGymSpaces + (Tuple, Dict)))
        for subspace in space.spaces
    ):
        return iter(items)

    return zip(
        *[iterate(subspace, items[i]) for i, subspace in enumerate(space.spaces)]
    )


@iterate.register(Dict)
def _iterate_dict(space, items):
    keys, values = zip(
        *[
            (key, iterate(subspace, items[key]))
            for key, subspace in space.spaces.items()
        ]
    )
    for item in zip(*values):
        yield OrderedDict([(key, value) for (key, value) in zip(keys, item)])


@iterate.register(Space)
def _iterate_custom(space, items):
    raise CustomSpaceError(
        f"Unable to iterate over {items}, since {space} "
        "is a custom `gym.Space` instance (i.e. not one of "
        "`Box`, `Dict`, etc...)."
    )
