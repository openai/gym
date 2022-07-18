"""Implementation of a space that represents the cartesian product of other spaces as a dictionary."""
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any
from typing import Dict as TypingDict
from typing import List, Optional
from typing import Sequence as TypingSequence
from typing import Tuple, Union

import numpy as np

from gym.spaces.space import Space
from gym.utils import seeding

if sys.version_info >= (3, 8):
    from typing import OrderedDict as TypingOrderedDict
else:
    from typing_extensions import OrderedDict as TypingOrderedDict


class Dict(Space[TypingDict[str, Space]], Mapping):
    """A dictionary of :class:`Space` instances.

    Elements of this space are (ordered) dictionaries of elements from the constituent spaces.

    Example usage:

        >>> from gym.spaces import Dict, Discrete
        >>> observation_space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
        >>> observation_space.sample()
        OrderedDict([('position', 1), ('velocity', 2)])

    Example usage [nested]::

        >>> from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
        >>> Dict(
        ...     {
        ...         "ext_controller": MultiDiscrete([5, 2, 2]),
        ...         "inner_state": Dict(
        ...             {
        ...                 "charge": Discrete(100),
        ...                 "system_checks": MultiBinary(10),
        ...                 "job_status": Dict(
        ...                     {
        ...                         "task": Discrete(5),
        ...                         "progress": Box(low=0, high=100, shape=()),
        ...                     }
        ...                 ),
        ...             }
        ...         ),
        ...     }
        ... )

    It can be convenient to use :class:`Dict` spaces if you want to make complex observations or actions more human-readable.
    Usually, it will be not be possible to use elements of this space directly in learning code. However, you can easily
    convert `Dict` observations to flat arrays by using a :class:`gym.wrappers.FlattenObservation` wrapper. Similar wrappers can be
    implemented to deal with :class:`Dict` actions.
    """

    def __init__(
        self,
        spaces: Optional[
            Union[
                TypingDict[str, Space],
                TypingOrderedDict[str, Space],
                TypingSequence[Tuple[str, Space]],
            ]
        ] = None,
        seed: Optional[Union[dict, int, seeding.RandomNumberGenerator]] = None,
        **spaces_kwargs: Space,
    ):
        """Constructor of :class:`Dict` space.

        This space can be instantiated in one of two ways: Either you pass a dictionary
        of spaces to :meth:`__init__` via the ``spaces`` argument, or you pass the spaces as separate
        keyword arguments (where you will need to avoid the keys ``spaces`` and ``seed``)

        Example::

            >>> from gym.spaces import Box, Discrete
            >>> Dict({"position": Box(-1, 1, shape=(2,)), "color": Discrete(3)})
            Dict(color:Discrete(3), position:Box(-1.0, 1.0, (2,), float32))
            >>> Dict(position=Box(-1, 1, shape=(2,)), color=Discrete(3))
            Dict(color:Discrete(3), position:Box(-1.0, 1.0, (2,), float32))

        Args:
            spaces: A dictionary of spaces. This specifies the structure of the :class:`Dict` space
            seed: Optionally, you can use this argument to seed the RNGs of the spaces that make up the :class:`Dict` space.
            **spaces_kwargs: If ``spaces`` is ``None``, you need to pass the constituent spaces as keyword arguments, as described above.
        """
        assert (spaces is None) != (
            len(spaces_kwargs) == 0
        ), "A Dict space cannot be empty and can only be initialised with a dictionary OR keywords."

        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
            try:
                spaces = OrderedDict(sorted(spaces.items()))
            except TypeError:  # raise when sort by different types of keys
                spaces = OrderedDict(spaces.items())
        elif isinstance(spaces, Sequence):
            spaces = OrderedDict(spaces)

        assert isinstance(
            spaces, OrderedDict
        ), f"Unexpected Dict space input, expecting dict, OrderedDict or Sequence, actual type: {type(spaces)}"

        self.spaces = spaces
        assert all(
            isinstance(space, Space) for space in spaces.values()
        ), "Dict space element is not an instance of Space: " + next(
            f"key={key}, space={space}"
            for key, space in spaces.items()
            if not isinstance(space, Space)
        )
        super().__init__(
            None, None, seed  # type: ignore
        )  # None for shape and dtype, since it'll require special handling

    def seed(
        self, seed: Optional[Union[TypingDict[str, int], int]] = None
    ) -> List[int]:
        """Seed the PRNG of this space and all subspaces."""
        seeds = []
        if isinstance(seed, dict):
            assert (
                seed.keys() == self.spaces.keys()
            ), f"The seed keys: {seed.keys()} are not identical to space keys: {self.spaces.keys()}"
            for key in seed.keys():
                seeds += self.spaces[key].seed(seed[key])
        elif isinstance(seed, int):
            seeds = super().seed(seed)
            # replace=False - unique subseed for each subspace, if you have more spaces than np.iinfo(int).max then something is terribly wrong
            subseeds = self.np_random.choice(
                np.iinfo(int).max,
                size=len(self.spaces),
                replace=False,
            )
            for subspace, subseed in zip(self.spaces.values(), subseeds):
                seeds += subspace.seed(int(subseed))
        elif seed is None:
            for space in self.spaces.values():
                seeds += space.seed(None)
        else:
            raise TypeError(
                f"Passed seed not of an expected type: dict or int or None, actual type: {type(seed)}"
            )

        return seeds

    def sample(self, mask: Optional[TypingDict[str, Any]] = None) -> dict:
        """Generates a single random sample from this space.

        The sample is an ordered dictionary of independent samples from the constituent spaces.

        Args:
            mask: An optional mask for each of the subspaces, expects the same keys as the space

        Returns:
            A dictionary with the same key and sampled values from :attr:`self.spaces`
        """
        if mask is not None:
            assert isinstance(
                mask, dict
            ), f"Expects mask to be a dict, actual type: {type(mask)}"
            assert (
                mask.keys() == self.spaces.keys()
            ), f"Expect mask keys to be same as space keys, mask keys: {mask.keys()}, space keys: {self.spaces.keys()}"
            return OrderedDict(
                [(k, space.sample(mask[k])) for k, space in self.spaces.items()]
            )

        return OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, dict):
            if x.keys() == self.spaces.keys():
                return all(x[key] in self.spaces[key] for key in self.spaces.keys())
        return False

    def __getitem__(self, key: str) -> Space:
        """Get the space that is associated to `key`."""
        return self.spaces[key]

    def __setitem__(self, key: str, value: Space):
        """Set the space that is associated to `key`."""
        self.spaces[key] = value

    def __iter__(self):
        """Iterator through the keys of the subspaces."""
        yield from self.spaces

    def __len__(self) -> int:
        """Gives the number of simpler spaces that make up the `Dict` space."""
        return len(self.spaces)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return "Dict(" + ", ".join([f"{k}: {s}" for k, s in self.spaces.items()]) + ")"

    def to_jsonable(self, sample_n: list) -> dict:
        """Convert a batch of samples from this space to a JSONable data type."""
        # serialize as dict-repr of vectors
        return {
            key: space.to_jsonable([sample[key] for sample in sample_n])
            for key, space in self.spaces.items()
        }

    def from_jsonable(self, sample_n: TypingDict[str, list]) -> List[OrderedDict]:
        """Convert a JSONable data type to a batch of samples from this space."""
        dict_of_list: TypingDict[str, list] = {
            key: space.from_jsonable(sample_n[key])
            for key, space in self.spaces.items()
        }

        n_elements = len(next(iter(dict_of_list.values())))
        result = [
            OrderedDict({key: value[n] for key, value in dict_of_list.items()})
            for n in range(n_elements)
        ]
        return result
