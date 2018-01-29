import gym
from gym import spaces, Space
import numpy as np

class MultiDiscrete(Space):
    def __init__(self, nvec):
        """
        nvec: vector of counts of each categorical variable
        """
        self.nvec = np.asarray(nvec, dtype=np.int32)
        assert self.nvec.ndim == 1, 'nvec should be a 1d array (or list) of ints'
        Space.__init__(self, (self.nvec.size,), np.int8)
    def sample(self):
        return (spaces.np_random.rand(self.nvec.size) * self.nvec).astype(self.dtype)
    def contains(self, x):
        return (x < self.nvec).all() and x.dtype.kind in 'ui'
    def to_jsonable(self, sample_n):
        return [sample.tolist() for sample in sample_n]
    def from_jsonable(self, sample_n):
        return np.array(sample_n)

