import gym
import numpy as np
from .space import Space


class MultiDiscrete(Space):
    def __init__(self, nvec):
        """
        nvec: vector of counts of each categorical variable
        """
        assert (np.array(nvec) > 0).all(), 'nvec (counts) have to be positive'
        self.nvec = np.asarray(nvec, dtype=np.uint32)

        super().__init__(self.nvec.shape, np.uint32)
        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        return (self.np_random.random_sample(self.nvec.shape) * self.nvec).astype(self.dtype)

    def contains(self, x):
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return (0 <= x).all() and (x < self.nvec).all()

    def to_jsonable(self, sample_n):
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        return np.array(sample_n)

    def __repr__(self):
        return "MultiDiscrete({})".format(self.nvec)

    def __eq__(self, other):
        return np.all(self.nvec == other.nvec)
