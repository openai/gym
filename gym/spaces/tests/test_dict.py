import numpy as np
import pytest
from gym.spaces import Tuple, Box, Discrete, Dict
from collections import OrderedDict

def test_dict_space():
    space = Dict(
        ('d1', Discrete(2)),
        ('b1', Box(0,1,1)),
        ('t1', Tuple([Discrete(3), Box(0,1,2)])))
    assert list(space.keys()) == ['d1', 'b1', 't1']
    assert all([isinstance(x, y) for x, y in zip(space.values(), [Discrete, Box, Tuple])])
    sample = space.sample()
    assert isinstance(sample, OrderedDict)
    assert space.contains(sample)
    sample['b1'][0] = 1.1
    assert not space.contains(sample)
    
