from gym import Space

class Tuple(Space):
    """
    A tuple (i.e., product) of simpler spaces

    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """
    def __init__(self, spaces):
        self.spaces = spaces

    def sample(self):
        return tuple([space.sample() for space in self.spaces])

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space,part) in zip(self.spaces,x))

    def __repr__(self):
        return "Tuple(" + ", ". join([str(s) for s in self.spaces]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as list-repr of tuple of vectors
        return [space.to_jsonable([sample[i] for sample in sample_n]) \
                for i, space in enumerate(self.spaces)]

    def from_jsonable(self, sample_n):
        return zip(*[space.from_jsonable(sample_n[i]) for i, space in enumerate(self.spaces)])
