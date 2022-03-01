from __future__ import annotations
from gym.spaces.space import Space, T_cov
from typing import Sequence, Type, Optional


class SpaceWrapper(Space[T_cov]):
    """Base class for wrapper around spaces."""

    def __init__(
        self,
        space: Space,
        shape: Optional[Sequence[int]] = None,
        dtype: Optional[Type | str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            shape=shape if shape is not None else space.shape,
            dtype=dtype or space.dtype,
            seed=seed,
        )
        self.space = space

    def __getattr__(self, attr: str):
        if attr.startswith("_"):
            raise AttributeError(f"Attempted to get missing private attribute '{attr}'")
        return getattr(self.space, attr)

    def seed(self, seed: Optional[int] = None):
        """Seed the rng of the wrapped space."""
        return self.space.seed(seed)

    def sample(self) -> T_cov:
        """Take a sample from the wrapped space."""
        return self.space.sample()

    def contains(self, x) -> bool:
        """Checks if the wrapped space contains this sample."""
        return self.space.contains(x)

    def __str__(self):
        return f"<{type(self).__name__}{self.space}>"

    def __repr__(self) -> str:
        return str(self)

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        return self.space.to_jsonable(sample_n)

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        return self.space.from_jsonable(sample_n)

