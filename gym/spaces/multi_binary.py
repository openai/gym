import gym
import numpy as np

class MultiBinary(gym.Space):
    def __init__(self, n):
        self.n = n
        gym.Space.__init__(self, (self.n,), np.int8)
    def sample(self):
        return gym.spaces.np_random.randint(low=0, high=2, size=self.n).astype(self.dtype)
    def contains(self, x):
        return ((x==0) | (x==1)).all()
    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()
    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]