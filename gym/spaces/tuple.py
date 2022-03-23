from collections.abc import Sequence
from typing import Iterable, List, Optional, Union
import numpy as np
from .space import Space


class Tuple(Space[tuple], Sequence):
    """
    A tuple (i.e., product) of simpler spaces

    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """

    def __init__(
        self, spaces: Iterable[Space], seed: Optional[Union[int, List[int]]] = None
    ):
        spaces = tuple(spaces)
        self.spaces = spaces
        for space in spaces:
            assert isinstance(
                space, Space
            ), "Elements of the tuple must be instances of gym.Space"
        super().__init__(None, None, seed)  # type: ignore

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> list:
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
        return tuple(space.sample() for space in self.spaces)

    def contains(self, x) -> bool:
        if isinstance(x, (list, np.ndarray)):
            x = tuple(x)  # Promote list and ndarray to tuple for contains check
        return (
            isinstance(x, tuple)
            and len(x) == len(self.spaces)
            and all(space.contains(part) for (space, part) in zip(self.spaces, x))
        )

    def __repr__(self) -> str:
        return "Tuple(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def to_jsonable(self, sample_n) -> list:
        # serialize as list-repr of tuple of vectors
        return [
            space.to_jsonable([sample[i] for sample in sample_n])
            for i, space in enumerate(self.spaces)
        ]

    def from_jsonable(self, sample_n) -> list:
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
        return self.spaces[index]

    def __len__(self) -> int:
        return len(self.spaces)

    def __eq__(self, other) -> bool:
        return isinstance(other, Tuple) and self.spaces == other.spaces
