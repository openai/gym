"""A set of utility functions for lambda wrappers."""
from functools import singledispatch
from typing import Any, Callable

import jumpy as jp

from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space


@singledispatch
def update_dtype(space: Space, args: FuncArgType, fn: Callable) -> Any:
    """Transform space dtype with the provided args."""


@update_dtype.register(Discrete)
@update_dtype.register(MultiBinary)
@update_dtype.register(MultiDiscrete)
def _update_dtype_discrete(space, args: FuncArgType, fn: Callable):
    return space


@update_dtype.register(Box)
def _update_dtype_box(space, args: jp.dtype, fn: Callable):
    if not args:
        return space

    return Box(
        space.low.astype(args),
        space.high.astype(args),
        shape=space.shape,
        dtype=args,
    )
