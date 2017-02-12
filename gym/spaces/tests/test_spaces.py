import json # note: ujson fails this test due to float equality
import numpy as np
import pytest
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete


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
