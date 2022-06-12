"""Implementation of a space that represents the cartesian product of `Discrete` spaces."""
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from gym import logger
from gym.spaces.discrete import Discrete
from gym.spaces.space import Space
from gym.utils import seeding


class MultiDiscrete(Space[np.ndarray]):
    """This represents the cartesian product of arbitrary :class:`Discrete` spaces.

    It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space.

    Note:
        Some environment wrappers assume a value of 0 always represents the NOOP action.

    e.g. Nintendo Game Controller - Can be conceptualized as 3 discrete action spaces:

    1. Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
    2. Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    3. Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

    It can be initialized as ``MultiDiscrete([ 5, 2, 2 ])``

    """

    def __init__(
        self,
        nvec: Union[np.ndarray, List[int]],
        dtype=np.int64,
        seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
    ):
        """Constructor of :class:`MultiDiscrete` space.

        The argument ``nvec`` will determine the number of values each categorical variable can take.

        Although this feature is rarely used, :class:`MultiDiscrete` spaces may also have several axes
        if ``nvec`` has several axes:

        Example::

            >> d = MultiDiscrete(np.array([[1, 2], [3, 4]]))
            >> d.sample()
            array([[0, 0],
                   [2, 3]])

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

    def sample(self) -> np.ndarray:
        """Generates a single random sample this space."""
        return (self.np_random.random(self.nvec.shape) * self.nvec).astype(self.dtype)

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return bool(x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all())

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
        subspace.np_random.bit_generator.state = self.np_random.bit_generator.state
        return subspace

    def __len__(self):
        """Gives the ``len`` of samples from this space."""
        if self.nvec.ndim >= 2:
            logger.warn("Get length of a multi-dimensional MultiDiscrete space.")
        return len(self.nvec)

    def __eq__(self, other):
        """Check whether ``other`` is equivalent to this instance."""
        return isinstance(other, MultiDiscrete) and np.all(self.nvec == other.nvec)
