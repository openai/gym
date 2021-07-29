import json  # note: ujson fails this test due to float equality
from copy import copy

import numpy as np
import pytest

from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict


@pytest.mark.parametrize(
    "space",
    [
        Discrete(3),
        Box(low=0.0, high=np.inf, shape=(2, 2)),
        Tuple([Discrete(5), Discrete(10)]),
        Tuple(
            [
                Discrete(5),
                Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32),
            ]
        ),
        Tuple((Discrete(5), Discrete(2), Discrete(2))),
        MultiDiscrete([2, 2, 100]),
        MultiBinary(10),
        Dict(
            {
                "position": Discrete(5),
                "velocity": Box(
                    low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32
                ),
            }
        ),
    ],
)
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


@pytest.mark.parametrize(
    "space",
    [
        Discrete(3),
        Box(low=np.array([-10, 0]), high=np.array([10, 10]), dtype=np.float32),
        Box(low=-np.inf, high=np.inf, shape=(1, 3)),
        Tuple([Discrete(5), Discrete(10)]),
        Tuple(
            [
                Discrete(5),
                Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32),
            ]
        ),
        Tuple((Discrete(5), Discrete(2), Discrete(2))),
        MultiDiscrete([2, 2, 100]),
        MultiBinary(6),
        Dict(
            {
                "position": Discrete(5),
                "velocity": Box(
                    low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32
                ),
            }
        ),
    ],
)
def test_equality(space):
    space1 = space
    space2 = copy(space)
    assert space1 == space2, "Expected {} to equal {}".format(space1, space2)


@pytest.mark.parametrize(
    "spaces",
    [
        (Discrete(3), Discrete(4)),
        (MultiDiscrete([2, 2, 100]), MultiDiscrete([2, 2, 8])),
        (MultiBinary(8), MultiBinary(7)),
        (
            Box(low=np.array([-10, 0]), high=np.array([10, 10]), dtype=np.float32),
            Box(low=np.array([-10, 0]), high=np.array([10, 9]), dtype=np.float32),
        ),
        (
            Box(low=-np.inf, high=0.0, shape=(2, 1)),
            Box(low=0.0, high=np.inf, shape=(2, 1)),
        ),
        (Tuple([Discrete(5), Discrete(10)]), Tuple([Discrete(1), Discrete(10)])),
        (Dict({"position": Discrete(5)}), Dict({"position": Discrete(4)})),
        (Dict({"position": Discrete(5)}), Dict({"speed": Discrete(5)})),
    ],
)
def test_inequality(spaces):
    space1, space2 = spaces
    assert space1 != space2, "Expected {} != {}".format(space1, space2)


@pytest.mark.parametrize(
    "space",
    [
        Discrete(5),
        Box(low=0, high=255, shape=(2,), dtype="uint8"),
        Box(low=-np.inf, high=np.inf, shape=(3, 3)),
        Box(low=1.0, high=np.inf, shape=(3, 3)),
        Box(low=-np.inf, high=2.0, shape=(3, 3)),
    ],
)
def test_sample(space):
    space.seed(0)
    n_trials = 100
    samples = np.array([space.sample() for _ in range(n_trials)])
    expected_mean = 0.0
    if isinstance(space, Box):
        if space.is_bounded():
            expected_mean = (space.high + space.low) / 2
        elif space.is_bounded("below"):
            expected_mean = 1 + space.low
        elif space.is_bounded("above"):
            expected_mean = -1 + space.high
        else:
            expected_mean = 0.0
    elif isinstance(space, Discrete):
        expected_mean = space.n / 2
    else:
        raise NotImplementedError
    np.testing.assert_allclose(expected_mean, samples.mean(), atol=3.0 * samples.std())


@pytest.mark.parametrize(
    "spaces",
    [
        (Discrete(5), MultiBinary(5)),
        (
            Box(low=np.array([-10, 0]), high=np.array([10, 10]), dtype=np.float32),
            MultiDiscrete([2, 2, 8]),
        ),
        (
            Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
        ),
        (Dict({"position": Discrete(5)}), Tuple([Discrete(5)])),
        (Dict({"position": Discrete(5)}), Discrete(5)),
        (Tuple((Discrete(5),)), Discrete(5)),
        (
            Box(low=np.array([-np.inf, 0.0]), high=np.array([0.0, np.inf])),
            Box(low=np.array([-np.inf, 1.0]), high=np.array([0.0, np.inf])),
        ),
    ],
)
def test_class_inequality(spaces):
    assert spaces[0] == spaces[0]
    assert spaces[1] == spaces[1]
    assert spaces[0] != spaces[1]
    assert spaces[1] != spaces[0]


@pytest.mark.parametrize(
    "space_fn",
    [
        lambda: Dict(space1="abc"),
        lambda: Dict({"space1": "abc"}),
        lambda: Tuple(["abc"]),
    ],
)
def test_bad_space_calls(space_fn):
    with pytest.raises(AssertionError):
        space_fn()
