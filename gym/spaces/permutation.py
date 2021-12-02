import numpy as np
from gym.spaces.space import Space


class Permutation(Space):
    def __init__(self, n, k, dtype=np.int64, seed=None):
        assert n >= k, "`n` has to be no less than `k`"
        self.n = n
        self.k = k
        self.nvec = np.array([n] * k, dtype=dtype)
        super().__init__(shape=self.nvec.shape, dtype=dtype, seed=seed)

    def sample(self):
        return self.np_random.permutation(self.nvec[0])[: len(self.nvec)].astype(
            self.dtype
        )

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        is_contained = (
            x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()
        )
        is_unique = np.unique(x).size == len(x)
        return is_unique and is_contained

    def to_jsonable(self, sample_n):
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        return np.array(sample_n)

    def __repr__(self):
        return f"Permutation({self.n}, {self.k})"

    def __len__(self):
        return len(self.nvec)

    def __eq__(self, other):
        return isinstance(other, Permutation) and np.all(self.nvec == other.nvec)
