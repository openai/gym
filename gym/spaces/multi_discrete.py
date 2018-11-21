import gym
import numpy as np
from .space import Space


class MultiDiscrete(Space):
    def __init__(self, nvec):
        """
        nvec: vector of counts of each categorical variable
        """
        self.nvec = np.asarray(nvec, dtype=np.int32)

        gym.Space.__init__(self, self.nvec.shape, np.int8)
        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        return (self.np_random.random_sample(self.nvec.shape) * self.nvec).astype(self.dtype)

    def contains(self, x):
        return (0 <= x).all() and (x < self.nvec).all() and x.dtype.kind in 'ui'

    def to_jsonable(self, sample_n):
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        return np.array(sample_n)

    def __repr__(self):
        return "MultiDiscrete({})".format(self.nvec)

    def __eq__(self, other):
        return np.all(self.nvec == other.nvec)
