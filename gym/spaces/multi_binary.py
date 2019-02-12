import numpy as np

from .space import Space


class MultiBinary(Space):
    def __init__(self, n):
        self.n = n
        super().__init__((self.n,), np.int8)

    def sample(self):
        return self.np_random.randint(low=0, high=2, size=self.n).astype(self.dtype)

    @property
    def flat_dim(self):
        return int(self.n)

    def flatten(self, x):
        return np.asarray(x).flatten()

    def unflatten(self, x):
        return np.asarray(x).reshape(self.shape)

    def contains(self, x):
        return ((x==0) | (x==1)).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "MultiBinary({})".format(self.n)

    def __eq__(self, other):
        return self.n == other.n
