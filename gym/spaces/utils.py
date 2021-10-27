from collections import OrderedDict
from functools import singledispatch, reduce
import numpy as np
import operator as op

from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
from gym.spaces import MultiBinary
from gym.spaces import Tuple
from gym.spaces import Dict


@singledispatch
def flatdim(space):
    """Return the number of dimensions a flattened equivalent of this space
    would have.

    Accepts a space and returns an integer. Raises ``NotImplementedError`` if
    the space is not defined in ``gym.spaces``.
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@flatdim.register(Box)
@flatdim.register(MultiBinary)
def flatdim_box_multibinary(space):
    return reduce(op.mul, space.shape, 1)


@flatdim.register(Discrete)
def flatdim_discrete(space):
    return int(space.n)


@flatdim.register(MultiDiscrete)
def flatdim_multidiscrete(space):
    return int(np.sum(space.nvec))


@flatdim.register(Tuple)
def flatdim_tuple(space):
    return sum([flatdim(s) for s in space.spaces])


@flatdim.register(Dict)
def flatdim_dict(space):
    return sum([flatdim(s) for s in space.spaces.values()])


@singledispatch
def flatten(space, x):
    """Flatten a data point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Accepts a space and a point from that space. Always returns a 1D array.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@flatten.register(Box)
@flatten.register(MultiBinary)
def flatten_box_multibinary(space, x):
    return np.asarray(x, dtype=space.dtype).flatten()


@flatten.register(Discrete)
def flatten_discrete(space, x):
    onehot = np.zeros(space.n, dtype=space.dtype)
    onehot[x] = 1
    return onehot


@flatten.register(MultiDiscrete)
def flatten_multidiscrete(space, x):
    offsets = np.zeros((space.nvec.size + 1,), dtype=space.dtype)
    offsets[1:] = np.cumsum(space.nvec.flatten())

    onehot = np.zeros((offsets[-1],), dtype=space.dtype)
    onehot[offsets[:-1] + x.flatten()] = 1
    return onehot


@flatten.register(Tuple)
def flatten_tuple(space, x):
    return np.concatenate([flatten(s, x_part) for x_part, s in zip(x, space.spaces)])


@flatten.register(Dict)
def flatten_dict(space, x):
    return np.concatenate([flatten(s, x[key]) for key, s in space.spaces.items()])


@singledispatch
def unflatten(space, x):
    """Unflatten a data point from a space.

    This reverses the transformation applied by ``flatten()``. You must ensure
    that the ``space`` argument is the same as for the ``flatten()`` call.

    Accepts a space and a flattened point. Returns a point with a structure
    that matches the space. Raises ``NotImplementedError`` if the space is not
    defined in ``gym.spaces``.
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@unflatten.register(Box)
@unflatten.register(MultiBinary)
def unflatten_box_multibinary(space, x):
    return np.asarray(x, dtype=space.dtype).reshape(space.shape)


@unflatten.register(Discrete)
def unflatten_discrete(space, x):
    return int(np.nonzero(x)[0][0])


@unflatten.register(MultiDiscrete)
def unflatten_multidiscrete(space, x):
    offsets = np.zeros((space.nvec.size + 1,), dtype=space.dtype)
    offsets[1:] = np.cumsum(space.nvec.flatten())

    (indices,) = np.nonzero(x)
    return np.asarray(indices - offsets[:-1], dtype=space.dtype).reshape(space.shape)


@unflatten.register(Tuple)
def unflatten_tuple(space, x):
    dims = np.asarray([flatdim(s) for s in space.spaces], dtype=np.int_)
    list_flattened = np.split(x, np.cumsum(dims[:-1]))
    return tuple(
        unflatten(s, flattened) for flattened, s in zip(list_flattened, space.spaces)
    )


@unflatten.register(Dict)
def unflatten_dict(space, x):
    dims = np.asarray([flatdim(s) for s in space.spaces.values()], dtype=np.int_)
    list_flattened = np.split(x, np.cumsum(dims[:-1]))
    return OrderedDict(
        [
            (key, unflatten(s, flattened))
            for flattened, (key, s) in zip(list_flattened, space.spaces.items())
        ]
    )


@singledispatch
def flatten_space(space):
    """Flatten a space into a single ``Box``.

    This is equivalent to ``flatten()``, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    ``flatdim(space)`` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.

    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.

    Example::

        >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(3, 4, 5)
        >>> flatten_space(box)
        Box(60,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

    Example that flattens a discrete space::

        >>> discrete = Discrete(5)
        >>> flatten_space(discrete)
        Box(5,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

    Example that recursively flattens a dict::

        >>> space = Dict({"position": Discrete(2),
        ...               "velocity": Box(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(6,)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@flatten_space.register(Box)
def flatten_space_box(space):
    return Box(space.low.flatten(), space.high.flatten(), dtype=space.dtype)


@flatten_space.register(Discrete)
@flatten_space.register(MultiBinary)
@flatten_space.register(MultiDiscrete)
def flatten_space_binary(space):
    return Box(low=0, high=1, shape=(flatdim(space),), dtype=space.dtype)


@flatten_space.register(Tuple)
def flatten_space_tuple(space):
    space = [flatten_space(s) for s in space.spaces]
    return Box(
        low=np.concatenate([s.low for s in space]),
        high=np.concatenate([s.high for s in space]),
        dtype=np.result_type(*[s.dtype for s in space]),
    )


@flatten_space.register(Dict)
def flatten_space_dict(space):
    space = [flatten_space(s) for s in space.spaces.values()]
    return Box(
        low=np.concatenate([s.low for s in space]),
        high=np.concatenate([s.high for s in space]),
        dtype=np.result_type(*[s.dtype for s in space]),
    )
