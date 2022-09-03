"""Implementation of a space that represents the cartesian product of `Discrete` spaces."""
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from gym import logger
from gym.spaces.discrete import Discrete
from gym.spaces.space import Space


class MultiDiscrete(Space[np.ndarray]):
    """This represents the cartesian product of arbitrary :class:`Discrete` spaces.

    It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space.

    Note:
        Some environment wrappers assume a value of 0 always represents the NOOP action.

    e.g. Nintendo Game Controller - Can be conceptualized as 3 discrete action spaces:

    1. Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
    2. Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    3. Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

    It can be initialized as ``MultiDiscrete([ 5, 2, 2 ])`` such that a sample might be ``array([3, 1, 0])``.

    Although this feature is rarely used, :class:`MultiDiscrete` spaces may also have several axes
    if ``nvec`` has several axes:

    Example::

        >> d = MultiDiscrete(np.array([[1, 2], [3, 4]]))
        >> d.sample()
        array([[0, 0],
               [2, 3]])
    """

    def __init__(
        self,
        nvec: Union[np.ndarray, list],
        dtype=np.int64,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        """Constructor of :class:`MultiDiscrete` space.

        The argument ``nvec`` will determine the number of values each categorical variable can take.

        Args:
            nvec: vector of counts of each categorical variable. This will usually be a list of integers. However,
                you may also pass a more complicated numpy array if you'd like the space to have several axes.
            dtype: This should be some kind of integer type.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
        """
        self.nvec = np.array(nvec, dtype=dtype, copy=True)
        assert (self.nvec > 0).all(), "nvec (counts) have to be positive"

        super().__init__(self.nvec.shape, dtype, seed)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Has stricter type than :class:`gym.Space` - never None."""
        return self._shape  # type: ignore

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def sample(self, mask: Optional[tuple] = None) -> np.ndarray:
        """Generates a single random sample this space.

        Args:
            mask: An optional mask for multi-discrete, expects tuples with a `np.ndarray` mask in the position of each
                action with shape `(n,)` where `n` is the number of actions and `dtype=np.int8`.
                Only mask values == 1 are possible to sample unless all mask values for an action are 0 then the default action 0 is sampled.

        Returns:
            An `np.ndarray` of shape `space.shape`
        """
        if mask is not None:

            def _apply_mask(
                sub_mask: Union[np.ndarray, tuple],
                sub_nvec: Union[np.ndarray, np.integer],
            ) -> Union[int, List[int]]:
                if isinstance(sub_nvec, np.ndarray):
                    assert isinstance(
                        sub_mask, tuple
                    ), f"Expects the mask to be a tuple for sub_nvec ({sub_nvec}), actual type: {type(sub_mask)}"
                    assert len(sub_mask) == len(
                        sub_nvec
                    ), f"Expects the mask length to be equal to the number of actions, mask length: {len(sub_mask)}, nvec length: {len(sub_nvec)}"
                    return [
                        _apply_mask(new_mask, new_nvec)
                        for new_mask, new_nvec in zip(sub_mask, sub_nvec)
                    ]
                else:
                    assert np.issubdtype(
                        type(sub_nvec), np.integer
                    ), f"Expects the sub_nvec to be an action, actually: {sub_nvec}, {type(sub_nvec)}"
                    assert isinstance(
                        sub_mask, np.ndarray
                    ), f"Expects the sub mask to be np.ndarray, actual type: {type(sub_mask)}"
                    assert (
                        len(sub_mask) == sub_nvec
                    ), f"Expects the mask length to be equal to the number of actions, mask length: {len(sub_mask)}, action: {sub_nvec}"
                    assert (
                        sub_mask.dtype == np.int8
                    ), f"Expects the mask dtype to be np.int8, actual dtype: {sub_mask.dtype}"

                    valid_action_mask = sub_mask == 1
                    assert np.all(
                        np.logical_or(sub_mask == 0, valid_action_mask)
                    ), f"Expects all masks values to 0 or 1, actual values: {sub_mask}"

                    if np.any(valid_action_mask):
                        return self.np_random.choice(np.where(valid_action_mask)[0])
                    else:
                        return 0

            return np.array(_apply_mask(mask, self.nvec), dtype=self.dtype)

        return (self.np_random.random(self.nvec.shape) * self.nvec).astype(self.dtype)

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check

        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return bool(
            isinstance(x, np.ndarray)
            and x.shape == self.shape
            and x.dtype != object
            and np.all(0 <= x)
            and np.all(x < self.nvec)
        )

    def to_jsonable(self, sample_n: Iterable[np.ndarray]):
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        return np.array(sample_n)

    def __repr__(self):
        """Gives a string representation of this space."""
        return f"MultiDiscrete({self.nvec})"

    def __getitem__(self, index):
        """Extract a subspace from this ``MultiDiscrete`` space."""
        nvec = self.nvec[index]
        if nvec.ndim == 0:
            subspace = Discrete(nvec)
        else:
            subspace = MultiDiscrete(nvec, self.dtype)  # type: ignore

        # you don't need to deepcopy as np random generator call replaces the state not the data
        subspace.np_random.bit_generator.state = self.np_random.bit_generator.state

        return subspace

    def __len__(self):
        """Gives the ``len`` of samples from this space."""
        if self.nvec.ndim >= 2:
            logger.warn(
                "Getting the length of a multi-dimensional MultiDiscrete space."
            )
        return len(self.nvec)

    def __eq__(self, other):
        """Check whether ``other`` is equivalent to this instance."""
        return isinstance(other, MultiDiscrete) and np.all(self.nvec == other.nvec)
