import numpy as np

from .space import Space


class Box(Space):
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).
    
    There are two common use cases:
    
    * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)
        
    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box(2,)

    """
    def __init__(self, low, high, shape=None, dtype=np.float32):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)

        if shape is None:
            assert low.shape == high.shape, 'box dimension mismatch. '
            self.shape = low.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high), 'box requires scalar bounds. '
            self.shape = tuple(shape)
            self.low = np.full(self.shape, low)
            self.high = np.full(self.shape, high)

        self.low = self.low.astype(self.dtype)
        self.high = self.high.astype(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = -np.inf < self.low
        self.bounded_above = np.inf > self.high

        super(Box, self).__init__(self.shape, self.dtype)

    def is_bounded(self, manner="both"):
        below = np.all(self.bounded_below)
        above = np.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        Generates a single random sample inside of the Box. 

        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:
        
        * [a, b] : uniform distribution 
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high if self.dtype.kind == 'f' \
                else self.high.astype('int64') + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded   = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below &  self.bounded_above
        low_bounded =  self.bounded_below & ~self.bounded_above
        bounded     =  self.bounded_below &  self.bounded_above
        

        # Vectorized sampling by interval type
        sample[unbounded] = self.np_random.normal(
                size=unbounded[unbounded].shape)

        sample[low_bounded] = self.np_random.exponential(
            size=low_bounded[low_bounded].shape) + self.low[low_bounded]
        
        sample[upp_bounded] = -self.np_random.exponential(
            size=upp_bounded[upp_bounded].shape) - self.high[upp_bounded]
        
        sample[bounded] = self.np_random.uniform(low=self.low[bounded], 
                                            high=high[bounded],
                                            size=bounded[bounded].shape)

        return sample.astype(self.dtype)
        
    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Box" + str(self.shape)

    def __eq__(self, other):
        return isinstance(other, Box) and (self.shape == other.shape) and np.allclose(self.low, other.low) and np.allclose(self.high, other.high)
