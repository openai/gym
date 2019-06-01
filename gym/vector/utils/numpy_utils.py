import numpy as np

from gym.spaces import Tuple, Dict
from gym.vector.utils.spaces import _BaseGymSpaces
from collections import OrderedDict

__all__ = ['concatenate', 'create_empty_array']

def concatenate(items, out, space):
    assert isinstance(items, (list, tuple))
    if isinstance(space, _BaseGymSpaces):
        return concatenate_basic(items, out, space)
    elif isinstance(space, Tuple):
        return concatenate_tuple(items, out, space)
    elif isinstance(space, Dict):
        return concatenate_dict(items, out, space)
    else:
        raise NotImplementedError()

def concatenate_basic(items, out, space):
    return np.stack(items, axis=0, out=out)

def concatenate_tuple(items, out, space):
    return tuple(concatenate([item[i] for item in items],
        out[i], subspace) for (i, subspace) in enumerate(space.spaces))

def concatenate_dict(items, out, space):
    return OrderedDict([(key, concatenate([item[key] for item in items],
        out[key], subspace)) for (key, subspace) in space.spaces.items()])


def create_empty_array(space, n=1, fn=np.empty):
    if isinstance(space, _BaseGymSpaces):
        return create_empty_array_basic(space, n=n, fn=fn)
    elif isinstance(space, Tuple):
        return create_empty_array_tuple(space, n=n, fn=fn)
    elif isinstance(space, Dict):
        return create_empty_array_dict(space, n=n, fn=fn)
    else:
        raise NotImplementedError()

def create_empty_array_basic(space, n=1, fn=np.empty):
    return fn((n,) + space.shape, dtype=space.dtype)

def create_empty_array_tuple(space, n=1, fn=np.empty):
    return tuple(create_empty_array(subspace, n=n, fn=fn)
        for subspace in space.spaces)

def create_empty_array_dict(space, n=1, fn=np.empty):
    return OrderedDict([(key, create_empty_array(subspace, n=n, fn=fn))
        for (key, subspace) in space.spaces.items()])
