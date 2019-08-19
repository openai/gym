import numpy as np

from gym.spaces import Tuple, Dict
from gym.vector.utils.spaces import _BaseGymSpaces
from collections import OrderedDict

__all__ = ['concatenate', 'create_empty_array']

def concatenate(items, out, space):
    """Concatenate multiple samples from space into a single object.

    Parameters
    ----------
    items : iterable of samples of `space`
        Samples to be concatenated.

    out : tuple, dict, or `np.ndarray`
        The output object. This object is a (possibly nested) numpy array.

    space : `gym.spaces.Space` instance
        Observation space of a single environment in the vectorized environment.

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
    >>> concatenate(items, out, space)
    array([[0.6348213 , 0.28607962, 0.60760117],
           [0.87383074, 0.192658  , 0.2148103 ]], dtype=float32)
    """
    assert isinstance(items, (list, tuple))
    if isinstance(space, _BaseGymSpaces):
        return concatenate_base(items, out, space)
    elif isinstance(space, Tuple):
        return concatenate_tuple(items, out, space)
    elif isinstance(space, Dict):
        return concatenate_dict(items, out, space)
    else:
        raise NotImplementedError()

def concatenate_base(items, out, space):
    return np.stack(items, axis=0, out=out)

def concatenate_tuple(items, out, space):
    return tuple(concatenate([item[i] for item in items],
        out[i], subspace) for (i, subspace) in enumerate(space.spaces))

def concatenate_dict(items, out, space):
    return OrderedDict([(key, concatenate([item[key] for item in items],
        out[key], subspace)) for (key, subspace) in space.spaces.items()])


def create_empty_array(space, n=1, fn=np.zeros):
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
    if isinstance(space, _BaseGymSpaces):
        return create_empty_array_base(space, n=n, fn=fn)
    elif isinstance(space, Tuple):
        return create_empty_array_tuple(space, n=n, fn=fn)
    elif isinstance(space, Dict):
        return create_empty_array_dict(space, n=n, fn=fn)
    else:
        raise NotImplementedError()

def create_empty_array_base(space, n=1, fn=np.zeros):
    shape = space.shape if (n is None) else (n,) + space.shape
    return fn(shape, dtype=space.dtype)

def create_empty_array_tuple(space, n=1, fn=np.zeros):
    return tuple(create_empty_array(subspace, n=n, fn=fn)
        for subspace in space.spaces)

def create_empty_array_dict(space, n=1, fn=np.zeros):
    return OrderedDict([(key, create_empty_array(subspace, n=n, fn=fn))
        for (key, subspace) in space.spaces.items()])
