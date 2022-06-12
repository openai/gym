"""Implementation of a space that represents the cartesian product of other spaces."""
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

from gym.spaces.space import Space
from gym.utils import seeding


class Tuple(Space[tuple], Sequence):
    """A tuple (more precisely: the cartesian product) of :class:`Space` instances.

    Elements of this space are tuples of elements of the constituent spaces.

    Example usage::

        >>> from gym.spaces import Box, Discrete
        >>> observation_space = Tuple((Discrete(2), Box(-1, 1, shape=(2,))))
        >>> observation_space.sample()
        (0, array([0.03633198, 0.42370757], dtype=float32))
    """

    def __init__(
        self,
        spaces: Iterable[Space],
        seed: Optional[Union[int, List[int], seeding.RandomNumberGenerator]] = None,
    ):
        r"""Constructor of :class:`Tuple` space.

        The generated instance will represent the cartesian product :math:`\text{spaces}[0] \times ... \times \text{spaces}[-1]`.

        Args:
            spaces (Iterable[Space]): The spaces that are involved in the cartesian product.
            seed: Optionally, you can use this argument to seed the RNGs of the ``spaces`` to ensure reproducible sampling.
        """
        self.spaces = tuple(spaces)
        for space in self.spaces:
            assert isinstance(
                space, Space
            ), "Elements of the tuple must be instances of gym.Space"
        super().__init__(None, None, seed)  # type: ignore

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> list:
        """Seed the PRNG of this space and all subspaces."""
        seeds = []

        if isinstance(seed, list):
            for i, space in enumerate(self.spaces):
                seeds += space.seed(seed[i])
        elif isinstance(seed, int):
            seeds = super().seed(seed)
            try:
                subseeds = self.np_random.choice(
                    np.iinfo(int).max,
                    size=len(self.spaces),
                    replace=False,  # unique subseed for each subspace
                )
            except ValueError:
                subseeds = self.np_random.choice(
                    np.iinfo(int).max,
                    size=len(self.spaces),
                    replace=True,  # we get more than INT_MAX subspaces
                )

            for subspace, subseed in zip(self.spaces, subseeds):
                seeds.append(subspace.seed(int(subseed))[0])
        elif seed is None:
            for space in self.spaces:
                seeds += space.seed(seed)
        else:
            raise TypeError("Passed seed not of an expected type: list or int or None")

        return seeds

    def sample(self) -> tuple:
        """Generates a single random sample inside this space.

        This method draws independent samples from the subspaces.

        Returns:
            Tuple of the subspace's samples
        """
        return tuple(space.sample() for space in self.spaces)

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, (list, np.ndarray)):
            x = tuple(x)  # Promote list and ndarray to tuple for contains check
        return (
            isinstance(x, tuple)
            and len(x) == len(self.spaces)
            and all(space.contains(part) for (space, part) in zip(self.spaces, x))
        )

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return "Tuple(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def to_jsonable(self, sample_n: Sequence) -> list:
        """Convert a batch of samples from this space to a JSONable data type."""
        # serialize as list-repr of tuple of vectors
        return [
            space.to_jsonable([sample[i] for sample in sample_n])
            for i, space in enumerate(self.spaces)
        ]

    def from_jsonable(self, sample_n) -> list:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [
            sample
            for sample in zip(
                *[
                    space.from_jsonable(sample_n[i])
                    for i, space in enumerate(self.spaces)
                ]
            )
        ]

    def __getitem__(self, index: int) -> Space:
        """Get the subspace at specific `index`."""
        return self.spaces[index]

    def __len__(self) -> int:
        """Get the number of subspaces that are involved in the cartesian product."""
        return len(self.spaces)

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return isinstance(other, Tuple) and self.spaces == other.spaces
