import json # note: ujson fails this test due to float equality
import numpy as np
import pytest
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary


@pytest.mark.parametrize("space", [
              Discrete(3),
              Tuple([Discrete(5), Discrete(10)]),
              Tuple([Discrete(5), Box(np.array([0,0]),np.array([1,5]))]),
              Tuple((Discrete(5), Discrete(2), Discrete(2))),
              MultiDiscrete([ [0, 1], [0, 1], [0, 100] ])
              ])
def test_roundtripping(space):
    sample_1 = space.sample()
    sample_2 = space.sample()
    assert space.contains(sample_1)
    assert space.contains(sample_2)
    json_rep = space.to_jsonable([sample_1, sample_2])

    json_roundtripped = json.loads(json.dumps(json_rep))

    samples_after_roundtrip = space.from_jsonable(json_roundtripped)
    sample_1_prime, sample_2_prime = samples_after_roundtrip

    s1 = space.to_jsonable([sample_1])
    s1p = space.to_jsonable([sample_1_prime])
    s2 = space.to_jsonable([sample_2])
    s2p = space.to_jsonable([sample_2_prime])
    assert s1 == s1p, "Expected {} to equal {}".format(s1, s1p)
    assert s2 == s2p, "Expected {} to equal {}".format(s2, s2p)

def test_space_eq():
  # Tuple Spaces
  ts1 = Tuple((Discrete(5), Discrete(8)))
  ts2 = Tuple([Discrete(5), Discrete(8)])
  ts3 = Tuple((Discrete(6), Discrete(8)))

  assert ts1 == ts2
  assert not (ts1 == ts3)

  # Discrete spaces
  ds1 = Discrete(5)
  mb1 = MultiBinary(5)
  ds2 = Discrete(5)
  mb2 = MultiBinary(5)
  assert not (ds1 == mb1)
  assert not (mb1 == ds1)
  assert ds1 == ds2
  assert mb1 == mb2

  md1 = MultiDiscrete(np.array([[0, 2], [0, 2]]))
  bs1 = Box(np.array([0, 0]), np.array([2, 2]))
  md2 = MultiDiscrete(np.array([[0, 2], [0, 2]]))
  bs2 = Box(np.array([0, 0]), np.array([2, 2]))
  assert not (md1 == bs1)
  assert not (bs1 == md1)
  assert md1 == md2
  assert bs1 == bs2