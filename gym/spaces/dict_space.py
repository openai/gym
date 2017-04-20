from gym import Space
import numpy as np
from collections import OrderedDict

class Dict(Space):
    """
    A ordered dict of simpler spaces

    Example usage:
    s = Dict(('b', Discrete(2)), ('a', Box(0,1,2)))
    s.sample()
    -> OrderedDict([('b', 0), ('a', array([ 0.59284462,  0.84426575]))])
    """
    def __init__(self, *spaces):
        """
        spaces(OrderedDict or pairs of key and Space): an OrderedDict or ordered (key, Space)
        """
        if spaces is ():
            self.spaces = OrderedDict()
        elif isinstance(spaces[0], OrderedDict):
            assert isinstance(spaces[0], Space)
            self.spaces = spaces[0]
        elif all(isinstance(x, (list, tuple)) for x in spaces):
            self.spaces = OrderedDict(spaces)
        else:
            assert false

    def sample(self):
        """
        random sample of the space
        return: OrderedDict of sampled data
        """
        return OrderedDict([(key, space.sample()) for key, space in self.spaces.items()])

    def contains(self, x):
        '''x(OrderedDict): OrderedDict of data'''
        assert isinstance(x, OrderedDict)
        return isinstance(x, OrderedDict) and all(
            self.spaces[key].contains(x[key]) for key in self.spaces.keys())

    def __repr__(self):
        return "Dict(" + ", ".join([str(x) for x in self.spaces]) + ")"

    def to_jsonable(self, sample_n):
        '''
        convert the list of sample data to jsonable object
        sample_n(list): List of sample data
        return JSONable object
        '''
        return [OrderedDict(
            [(key, space.to_one_jsonable(sample[key])) for key, space in self.spaces.items()]) for sample in sample_n]

    def from_jsonable(self, sample_n):
        '''
        convert the jsonable object to the list of sample data
        sample_n(list): JSONable representation of sample data
        return List of sample data(OrderedDict)
        '''
        return [OrderedDict(
            [(key, space.from_one_jsonable(sample[key])) for key, space in self.spaces.items()]) for sample in sample_n]

    def to_one_jsonable(self, sample):
        '''
        convert a sample data to jsonable object
        sample(object): a sample of data
        return JSONable object
        '''
        return OrderedDict([(key, space.to_one_jsonable(sample[key])) for key, space in self.spaces.items()])

    def __getitem__(self, key):
        return self.spaces[key]

    def __setitem__(self, key, value):
        '''
        Setter of Dict space. Note the assignment order is reserved.
        d = Dict()
        d['b'] = Discrete(3)
        d['a'] = Box(0,1,2)
        d.keys() --> ['b', 'a']
        '''
        assert isinstance(value, Space)
        self.spaces[key] = value

    def __iter__(self):
        for key, value in self.spaces.items():
            yield key, value

    def keys(self):
        return self.spaces.keys()

    def values(self):
        return self.spaces.values()

    def items(self):
        return self.spaces.items()
