import numpy as np

import gym
from gym.spaces import prng

class Box(gym.Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.

    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    def __init__(self, low, high, shape=None, named=None):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)
        if named:
            # Probably not a good idea to do this for high-dimensional spaces
            assert self.shape == (len(named),)
        self.named_dimensions = named
    def sample(self):
        return prng.np_random.uniform(low=self.low, high=self.high, size=self.low.shape)
    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()
    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    @property
    def shape(self):
        return self.low.shape
    def __repr__(self):
        # Don't try to show high/low/name for each dimension of high-dimensional spaces
        if len(self.shape) > 1 or self.shape[0] > 10:
            return "Box" + str(self.shape)
        range_strs = ["{:.2g}..{:.2g}".format(self.low[i], self.high[i]) for i in range(self.shape[0])]
        if self.named_dimensions:
            dimen_strs = ["{}: {}".format(self.named_dimensions[i], range_strs[i])
                for i in range(self.shape[0])]
        else:
            dimen_strs = range_strs
        return "Box({{{}}})".format(", ".join(dimen_strs))
    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)
