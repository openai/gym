from __future__ import annotations

from typing import (
    TypeVar,
    Generic,
    Optional,
    Sequence,
    Iterable,
    Mapping,
    Type,
)

import numpy as np

from gym.utils import seeding


T_cov = TypeVar("T_cov", covariant=True)


class Space(Generic[T_cov]):
    """Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.

    WARNING - Custom observation & action spaces can inherit from the `Space`
    class. However, most use-cases should be covered by the existing space
    classes (e.g. `Box`, `Discrete`, etc...), and container classes (`Tuple` &
    `Dict`). Note that parametrized probability distributions (through the
    `sample()` method), and batching functions (in `gym.vector.VectorEnv`), are
    only well-defined for instances of spaces provided in gym by default.
    Moreover, some implementations of Reinforcement Learning algorithms might
    not handle custom spaces properly. Use custom spaces with care.
    """

    def __init__(
        self,
        shape: Optional[Sequence[int]] = None,
        dtype: Optional[Type | str] = None,
        seed: Optional[int] = None,
    ):
        import numpy as np  # noqa ## takes about 300-400ms to import, so we load lazily

        self._shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._np_random = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> seeding.RandomNumberGenerator:
        """Lazily seed the rng since this is expensive and only needed if
        sampling from this space.
        """
        if self._np_random is None:
            self.seed()

        return self._np_random  # type: ignore  ## self.seed() call guarantees right type.

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """Return the shape of the space as an immutable property"""
        return self._shape

    def sample(self) -> T_cov:
        """Randomly sample an element of this space. Can be
        uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> list:
        """Seed the PRNG of this space."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def contains(self, x) -> bool:
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def __contains__(self, x) -> bool:
        return self.contains(x)

    def __setstate__(self, state: Iterable | Mapping):
        # Don't mutate the original state
        state = dict(state)

        # Allow for loading of legacy states.
        # See:
        #   https://github.com/openai/gym/pull/2397 -- shape
        #   https://github.com/openai/gym/pull/1913 -- np_random
        #
        if "shape" in state:
            state["_shape"] = state["shape"]
            del state["shape"]
        if "np_random" in state:
            state["_np_random"] = state["np_random"]
            del state["np_random"]

        # Update our state
        self.__dict__.update(state)

    def to_jsonable(self, sample_n: Sequence[T_cov]) -> list:
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return list(sample_n)

    def from_jsonable(self, sample_n: list) -> list[T_cov]:
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n
