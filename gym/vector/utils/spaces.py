import numpy as np
from collections import OrderedDict
from functools import singledispatch

from gym import spaces
from gym.spaces import Space, Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict

_BaseGymSpaces = (Box, Discrete, MultiDiscrete, MultiBinary)
__all__ = ["_BaseGymSpaces", "batch_space"]



@singledispatch
def batch_space(space: Space, n: int = 1) -> Space:
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
    raise ValueError('Cannot batch space with type `{0}`. The space must '
                     'be a valid `gym.Space` instance.'.format(type(space)))


@batch_space.register(Box)
def batch_space_Box(space: Box, n: int = 1) -> Box:
    repeats = tuple([n] + [1] * space.low.ndim)
    low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
    return Box(low=low, high=high, dtype=space.dtype)


@batch_space.register(Discrete)
def batch_space_Discrete(space: Discrete, n: int = 1) -> MultiDiscrete:
    return MultiDiscrete(np.full((n,), space.n, dtype=space.dtype))


@batch_space.register(MultiDiscrete)
def batch_space_multi_discrete(space: MultiDiscrete, n: int = 1) -> Box:
    repeats = tuple([n] + [1] * space.nvec.ndim)
    high = np.tile(space.nvec, repeats) - 1
    return Box(low=np.zeros_like(high), high=high, dtype=space.dtype)


@batch_space.register(MultiBinary)
def batch_space_multi_binary(space: MultiBinary, n: int = 1) -> Box:
    return Box(low=0, high=1, shape=(n,) + space.shape, dtype=space.dtype)


@batch_space.register(spaces.Tuple)
def batch_space_tuple(space: spaces.Tuple, n: int = 1) -> spaces.Tuple:
    return Tuple(tuple(batch_space(subspace, n=n) for subspace in space.spaces))


@batch_space.register(spaces.Dict)
def batch_space_dict(space: spaces.Dict, n: int = 1) -> spaces.Dict:
    return Dict(OrderedDict([(key, batch_space(subspace, n=n))
        for (key, subspace) in space.spaces.items()]))


@batch_space.register(Space)
def batch_space_custom(space: Space, n: int = 1) -> Space:
    return spaces.Tuple(tuple(space for _ in range(n)))
