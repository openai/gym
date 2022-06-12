"""Implementation of a space consisting of finitely many elements."""
from typing import Optional, Union

import numpy as np

from gym.spaces.space import Space
from gym.utils import seeding


class Discrete(Space[int]):
    r"""A space consisting of finitely many elements.

    This class represents a finite subset of integers, more specifically a set of the form :math:`\{ a, a+1, \dots, a+n-1 \}`.

    Example::

        >>> Discrete(2)            # {0, 1}
        >>> Discrete(3, start=-1)  # {-1, 0, 1}
    """

    def __init__(
        self,
        n: int,
        seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
        start: int = 0,
    ):
        r"""Constructor of :class:`Discrete` space.

        This will construct the space :math:`\{\text{start}, ..., \text{start} + n - 1\}`.

        Args:
            n (int): The number of elements of this space.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the ``Dict`` space.
            start (int): The smallest element of this space.
        """
        assert isinstance(n, (int, np.integer))
        assert n > 0, "n (counts) have to be positive"
        assert isinstance(start, (int, np.integer))
        self.n = int(n)
        self.start = int(start)
        super().__init__((), np.int64, seed)

    def sample(self) -> int:
        """Generates a single random sample from this space.

        A sample will be chosen uniformly at random.

        Returns:
            A sampled integer from the space
        """
        return int(self.start + self.np_random.integers(self.n))

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)  # type: ignore
        else:
            return False
        return self.start <= as_int < self.start + self.n

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        if self.start != 0:
            return "Discrete(%d, start=%d)" % (self.n, self.start)
        return "Discrete(%d)" % self.n

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )

    def __setstate__(self, state):
        """Used when loading a pickled space.

        This method has to be implemented explicitly to allow for loading of legacy states.

        Args:
            state: The new state
        """
        super().__setstate__(state)

        # Don't mutate the original state
        state = dict(state)

        # Allow for loading of legacy states.
        # See https://github.com/openai/gym/pull/2470
        if "start" not in state:
            state["start"] = 0

        # Update our state
        self.__dict__.update(state)
