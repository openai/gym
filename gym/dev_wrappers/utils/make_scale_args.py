"""A set of utility functions for lambda wrappers."""
from copy import deepcopy
from functools import singledispatch
from typing import Callable
from typing import Dict as TypingDict
from typing import Sequence
from typing import Tuple as TypingTuple

from gym.dev_wrappers import ArgType, FuncArgType
from gym.spaces import Box, Dict, Space, Tuple


@singledispatch
def make_scale_args(space: Space, args: FuncArgType, fn: Callable):
    """Compute args for rescaling action function.

    Action space args needs to be extended in order
    to correctly rescale the actions.
    i.e. args before: {"body":{"left_arm": (-0.5,0.5)}, ...}
    args after: {"body":{"left_arm": (-0.5,0.5,-1,1)}, ...}
    where -1, 1 was the old action space bound.
    old action space is needed to rescale actions.
    """
    ...


@make_scale_args.register(Box)
def _make_scale_args_box(space: Space, args: Sequence, fn: Callable):
    if args is None:
        return (space.low, space.high, space.low, space.high)
    return (*args, space.low, space.high)


@make_scale_args.register(Dict)
def _make_scale_args_dict(
    space: Tuple, args: FuncArgType[TypingDict[str, ArgType]], fn: Callable
):
    extended_args = deepcopy(args)

    for arg in args:
        extended_args[arg] = fn(space[arg], args[arg], fn)

    return extended_args


@make_scale_args.register(Tuple)
def _make_scale_args_tuple(
    space: Tuple, args: FuncArgType[TypingTuple[float, float]], fn: Callable
):
    assert isinstance(args, Sequence)
    assert len(space) == len(args)

    extended_args = [arg for arg in args]

    for i in range(len(args)):
        extended_args[i] = fn(space[i], args[i], fn)
    return extended_args
