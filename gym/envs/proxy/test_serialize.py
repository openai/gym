"""
Tests for zmq_serialize
"""
import sys
import numpy as np
from gym import error
from gym.spaces import Box,Discrete,Tuple
from gym.envs.proxy import zmq_serialize

def serdes(o):
    parts = zmq_serialize.dump_msg(o)
    # touch all the bytes
    parts = [('blub'+x)[4:] for x in parts]
    o2 = zmq_serialize.load_msg(parts)
    return o2

# Spaces
def test_spaces_Discrete():
    a = Discrete(17)
    b = serdes(a)
    assert str(a) == str(b)
    assert a.n == b.n

def test_spaces_Tuple():
    a = Tuple([Discrete(17), Box(low=np.array([-1,-1]), high=np.array([+1, +1]))])
    b = serdes(a)
    assert str(a) == str(b)
    assert b.contains(a.sample())
    assert a.contains(b.sample())

def test_spaces_Tuple():
    a = Tuple([Discrete(17), Box(low=np.array([-1,-1]), high=np.array([+1, +1]))])
    b = serdes(a)
    assert str(a) == str(b)
    assert b.contains(a.sample())
    assert a.contains(b.sample())


# Numpy array
def test_numpy_uint8():
    a = np.array([[1,2,3], [4,5,6]], dtype='uint8')
    b = serdes(a)
    assert str(a) == str(b)
    assert np.all(a==b)

def test_numpy_float64():
    a = np.array([[1,2,3], [4,5,6]], dtype='float64')
    b = serdes(a)
    assert str(a) == str(b)
    assert np.all(a==b)

def test_spaces_Box():
    a = Box(low=np.array([-1,-1]), high=np.array([+1, +1]))
    b = serdes(a)
    assert str(a) == str(b)
    assert np.all(a.low == b.low)
    assert np.all(a.high == b.high)

def test_numpy_img():
    """
    Takes about 0.3 seconds on a Macbook Pro (Mid-2015).
    If it takes much longer, something might be broken
    """
    a = np.random.randint(low=0, high=256, size=[640,480,3], dtype='uint8')
    for i in range(1000):
        b = serdes(a)
        assert np.all(a==b)

def test_numpy_random_floats():
    a = np.random.random(size=[1000])
    for i in range(10):
        b = serdes(a)
        assert np.all(a==b)
