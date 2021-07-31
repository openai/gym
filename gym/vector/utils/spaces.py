import numpy as np
from collections import OrderedDict

from gym.spaces import Space, Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict
from gym.error import CustomSpaceError

_BaseGymSpaces = (Box, Discrete, MultiDiscrete, MultiBinary)
__all__ = ["_BaseGymSpaces", "batch_space", "iterate"]


def batch_space(space, n=1):
    """Create a (batched) space, containing multiple copies of a single space.

    Parameters
    ----------
    space : `gym.spaces.Space` instance
        Space (e.g. the observation space) for a single environment in the
        vectorized environment.

    n : int
        Number of environments in the vectorized environment.

    Returns
    -------
    batched_space : `gym.spaces.Space` instance
        Space (e.g. the observation space) for a batch of environments in the
        vectorized environment.

    Example
    -------
    >>> from gym.spaces import Box, Dict
    >>> space = Dict({
    ... 'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
    ... 'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)})
    >>> batch_space(space, n=5)
    Dict(position:Box(5, 3), velocity:Box(5, 2))
    """
    if isinstance(space, _BaseGymSpaces):
        return batch_space_base(space, n=n)
    elif isinstance(space, Tuple):
        return batch_space_tuple(space, n=n)
    elif isinstance(space, Dict):
        return batch_space_dict(space, n=n)
    elif isinstance(space, Space):
        return batch_space_custom(space, n=n)
    else:
        raise ValueError(
            f"Cannot batch space with type `{type(space)}`. The space must be a valid `gym.Space` instance."
        )


def batch_space_base(space, n=1):
    if isinstance(space, Box):
        repeats = tuple([n] + [1] * space.low.ndim)
        low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
        return Box(low=low, high=high, dtype=space.dtype)

    elif isinstance(space, Discrete):
        return MultiDiscrete(np.full((n,), space.n, dtype=space.dtype))

    elif isinstance(space, MultiDiscrete):
        repeats = tuple([n] + [1] * space.nvec.ndim)
        high = np.tile(space.nvec, repeats) - 1
        return Box(low=np.zeros_like(high), high=high, dtype=space.dtype)

    elif isinstance(space, MultiBinary):
        return Box(low=0, high=1, shape=(n,) + space.shape, dtype=space.dtype)

    else:
        raise ValueError(f"Space type `{type(space)}` is not supported.")


def batch_space_tuple(space, n=1):
    return Tuple(tuple(batch_space(subspace, n=n) for subspace in space.spaces))


def batch_space_dict(space, n=1):
    return Dict(
        OrderedDict(
            [
                (key, batch_space(subspace, n=n))
                for (key, subspace) in space.spaces.items()
            ]
        )
    )


def batch_space_custom(space, n=1):
    return Tuple(tuple(space for _ in range(n)))


def iterate(items, space):
    """Iterate over the elements of a (batched) space.

    Parameters
    ----------
    items : samples of `space`
        Items to be iterated over.

    space : `gym.spaces.Space` instance
        Space to which `items` belong to.

    Returns
    -------
    iterator : `Iterable` instance
        Iterator over the elements in `items`.

    Example
    -------
    >>> from gym.spaces import Box, Dict
    >>> space = Dict({
    ... 'position': Box(low=0, high=1, shape=(2, 3), dtype=np.float32),
    ... 'velocity': Box(low=0, high=1, shape=(2, 2), dtype=np.float32)})
    >>> items = space.sample()
    >>> it = iterate(items, space)
    >>> next(it)
    {'position': array([-0.99644893, -0.08304597, -0.7238421 ], dtype=float32),
    'velocity': array([0.35848552, 0.1533453 ], dtype=float32)}
    >>> next(it)
    {'position': array([-0.67958736, -0.49076623,  0.38661423], dtype=float32),
    'velocity': array([0.7975036 , 0.93317133], dtype=float32)}
    >>> next(it)
    StopIteration
    """
    if isinstance(space, _BaseGymSpaces):
        return iterate_base(items, space)
    elif isinstance(space, Tuple):
        return iterate_tuple(items, space)
    elif isinstance(space, Dict):
        return iterate_dict(items, space)
    elif isinstance(space, Space):
        return iterate_custom(items, space)
    else:
        raise ValueError(
            "Space of type `{0}` is not a valid `gym.Space` "
            "instance.".format(type(space))
        )


def iterate_base(items, space):
    if isinstance(space, Discrete):
        raise TypeError("Unable to iterate over a space of type `Discrete`.")
    try:
        return iter(items)
    except TypeError:
        raise TypeError(f"Unable to iterate over the following elements: {items}")


def iterate_tuple(items, space):
    # If this is a tuple of custom subspaces only, then simply iterate over items
    if all(
        not isinstance(subspace, (_BaseGymSpaces, Tuple, Dict))
        for subspace in space.spaces
    ):
        return iter(items)

    return zip(
        *[iterate(items[i], subspace) for i, subspace in enumerate(space.spaces)]
    )


def iterate_dict(items, space):
    keys, values = zip(
        *[
            (key, iterate(items[key], subspace))
            for key, subspace in space.spaces.items()
        ]
    )
    for item in zip(*values):
        yield OrderedDict([(key, value) for (key, value) in zip(keys, item)])


def iterate_custom(items, space):
    raise CustomSpaceError(
        f"Unable to iterate over {items}, since {space} "
        "is a custom `gym.Space` instance (i.e. not one of "
        "`Box`, `Dict`, etc...)."
    )
