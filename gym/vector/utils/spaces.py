import numpy as np
from collections import OrderedDict

from gym.spaces import Space, Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict

_BaseGymSpaces = (Box, Discrete, MultiDiscrete, MultiBinary)
__all__ = ['_BaseGymSpaces', 'batch_space']

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
        raise ValueError('Cannot batch space with type `{0}`. The space must '
                         'be a valid `gym.Space` instance.'.format(type(space)))

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
        raise ValueError('Space type `{0}` is not supported.'.format(type(space)))

def batch_space_tuple(space, n=1):
    return Tuple(tuple(batch_space(subspace, n=n) for subspace in space.spaces))

def batch_space_dict(space, n=1):
    return Dict(OrderedDict([(key, batch_space(subspace, n=n))
        for (key, subspace) in space.spaces.items()]))

def batch_space_custom(space, n=1):
    return Tuple(tuple(space for _ in range(n)))
