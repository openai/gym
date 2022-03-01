from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, List, Sequence, TypeVar

import numpy as np
from gym.spaces.box import Box
from gym.spaces.dict import Dict as DictSpace
from gym.spaces.discrete import Discrete
from gym.spaces.multi_binary import MultiBinary
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.other.wrapper import SpaceWrapper
from gym.spaces.space import Space
from gym.spaces.tuple import Tuple as TupleSpace

T = TypeVar("T")
T_cot = TypeVar("T_cot", contravariant=True)
T_cov = TypeVar("T_cov", covariant=True)
Predicate = Callable[[T_cot], bool]


class FilteredSpace(SpaceWrapper[T_cov]):
    """Adds assumptions (predicates) to a space, limiting the samples that are considered valid.
    
    Uses rejection sampling with a maximum of `max_sampling_attempts` attempts.

    >>> from gym.spaces import Discrete, Box, Space
    >>> import numpy as np

    >>> odd_digits = Discrete(10, seed=123).where(lambda n: n % 2 == 1)
    >>> 1 in odd_digits
    True
    >>> 2 in odd_digits
    False
    >>> odd_digits.sample()
    5

    Other cool examples:

    >>> unit_circle = Box(-1, 1, shape=(2,), dtype=np.float64, seed=123).where(lambda xy: (xy ** 2).sum() <= 1)
    >>> np.array((0.1, 0.2)) in unit_circle
    True
    >>> np.array((1, 1)) in unit_circle
    False
    >>> unit_circle.sample()
    array([ 0.36470373, -0.89235796])

    >>> def hypersphere(radius: float, dimensions: int) -> FilteredSpace[np.ndarray]:
    ...     return Box(-radius, +radius, shape=(dimensions,)).where(
    ...         lambda v: v.pow(2).sum() <= radius**dimensions
    ...     )
    ...
    >>> unit_sphere = hypersphere(radius=1, dimensions=3)
    """

    def __init__(
        self,
        space: Space[T_cov],
        predicates: Sequence[Predicate[T_cov]],
        max_sampling_attempts: int | None = None,
    ):
        super().__init__(space=space)
        self.space = space
        self.predicates = list(predicates)
        self.max_sampling_attempts = max_sampling_attempts

    def contains(self, x) -> bool:
        """Checks if the wrapped space contains this sample and if it fits all assumptions."""
        return self.space.contains(x) and all(
            predicate(x) for predicate in self.predicates
        )

    def sample(self) -> T_cov:
        sample = self.space.sample()
        attempts = 1
        while sample not in self:
            sample = self.space.sample()
            if self.max_sampling_attempts and attempts == self.max_sampling_attempts:
                raise RuntimeError(
                    f"Unable to find a valid sample with {self.max_sampling_attempts} attempts."
                )
            attempts += 1
        return sample

    def where(self, *predicates: Predicate[T_cov]) -> FilteredSpace[T_cov]:
        cls = type(self)
        return cls(self.space, predicates=self.predicates + list(predicates))

    def __repr__(self):
        return repr(self.space) + ".where(" + repr(self.predicates) + ")"


from gym.vector.utils.numpy_utils import concatenate, create_empty_array
from gym.vector.utils.shared_memory import (
    create_shared_memory,
    mp,
    read_from_shared_memory,
    write_to_shared_memory,
)
from gym.vector.utils.spaces import batch_space, iterate


def _batched_predicate(
    batched_sample: T, batched_space: Space[T], predicate: Predicate
) -> bool:
    """Predicate that applies `predicate` to all the items in `batched_sample`. Returns all(results)
    """
    return all(map(predicate, iterate(batched_space, batched_sample)))


@batch_space.register(FilteredSpace)
def _batch_filtered(space: FilteredSpace, n: int) -> Space | FilteredSpace:
    # NOTE: Depending on the type of the wrapped space, this could do something a bit different
    # (e.g. batch the predicate function too).

    if isinstance(
        space.space, (Box, Discrete, MultiDiscrete, MultiBinary, DictSpace, TupleSpace)
    ):
        # Create a predicate that will be applied on all the samples in the batch.
        batched_space = batch_space(space.space, n=n)
        batched_predicates: List[Predicate] = [
            functools.partial(
                _batched_predicate, batched_space=batched_space, predicate=predicate
            )
            for predicate in space.predicates
        ]
        return type(space)(space=batched_space, predicates=batched_predicates)

    # Just use the default behaviour in the worst case (Tuple of FilteredSpaces).
    return batch_space.dispatch(Space)(space, n=n)
    # NOTE: Is there a deepcopy missing from the underlying implementation?
    # return TupleSpace(tuple(copy.deepcopy(space) for _ in range(n)))


@iterate.register
def _iterate_filtered(space: FilteredSpace, items: Any):
    return iterate(space.space, items)


@concatenate.register
def _concatenate_filtered(space: FilteredSpace, items: Any, out: Any):
    return concatenate(space.space, items, out)


@create_empty_array.register
def _create_empty_filtered(
    space: FilteredSpace, n: int = 1, fn: Callable[..., np.ndarray] = np.zeros
):
    return create_empty_array(space.space, n=n, fn=fn)


@create_shared_memory.register
def _create_shared_memory_filtered(space: FilteredSpace, n: int = 1, ctx=mp):
    return create_shared_memory(space.space, n=n, ctx=ctx)


@write_to_shared_memory.register(FilteredSpace)
def _write_to_shared_memory_filtered(
    space: FilteredSpace[T], index: int, value: T, shared_memory: Any
) -> None:
    write_to_shared_memory(
        space.space, index=index, value=value, shared_memory=shared_memory
    )


@read_from_shared_memory.register
def _read_from_shared_memory_filtered(
    space: FilteredSpace, shared_memory: Any, n: int = 1
):
    return read_from_shared_memory(space.space, shared_memory=shared_memory, n=n)
