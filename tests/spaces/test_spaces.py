import json  # note: ujson fails this test due to float equality
import copy

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
    space2 = copy.copy(space)
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


def test_seed_Dict():
    test_space = Dict(
        {
            "a": Box(low=0, high=1, shape=(3, 3)),
            "b": Dict(
                {
                    "b_1": Box(low=-100, high=100, shape=(2,)),
                    "b_2": Box(low=-1, high=1, shape=(2,)),
                }
            ),
            "c": Discrete(5),
        }
    )

    seed_dict = {
        "a": 0,
        "b": {
            "b_1": 1,
            "b_2": 2,
        },
        "c": 3,
    }

    test_space.seed(seed_dict)

    # "Unpack" the dict sub-spaces into individual spaces
    a = Box(low=0, high=1, shape=(3, 3))
    a.seed(0)
    b_1 = Box(low=-100, high=100, shape=(2,))
    b_1.seed(1)
    b_2 = Box(low=-1, high=1, shape=(2,))
    b_2.seed(2)
    c = Discrete(5)
    c.seed(3)

    for i in range(10):
        test_s = test_space.sample()
        a_s = a.sample()
        assert (test_s["a"] == a_s).all()
        b_1_s = b_1.sample()
        assert (test_s["b"]["b_1"] == b_1_s).all()
        b_2_s = b_2.sample()
        assert (test_s["b"]["b_2"] == b_2_s).all()
        c_s = c.sample()
        assert test_s["c"] == c_s


def test_box_dtype_check():
    # Related Issues:
    # https://github.com/openai/gym/issues/2357
    # https://github.com/openai/gym/issues/2298

    space = Box(0, 2, tuple(), dtype=np.float32)

    # casting will match the correct type
    assert space.contains(0.5)

    # float64 is not in float32 space
    assert not space.contains(np.array(0.5))
    assert not space.contains(np.array(1))


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
def test_seed_returns_list(space):
    def assert_integer_list(seed):
        assert isinstance(seed, list)
        assert len(seed) >= 1
        assert all([isinstance(s, int) for s in seed])

    assert_integer_list(space.seed(None))
    assert_integer_list(space.seed(0))


def convert_sample_hashable(sample):
    if isinstance(sample, np.ndarray):
        return tuple(sample.tolist())
    if isinstance(sample, (list, tuple)):
        return tuple(convert_sample_hashable(s) for s in sample)
    if isinstance(sample, dict):
        return tuple(
            (key, convert_sample_hashable(value)) for key, value in sample.items()
        )

    return sample


def sample_equal(sample1, sample2):
    return convert_sample_hashable(sample1) == convert_sample_hashable(sample2)


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
def test_seed_reproducibility(space):
    space1 = space
    space2 = copy.deepcopy(space)

    space1.seed(None)
    space2.seed(None)

    assert space1.seed(0) == space2.seed(0)
    assert sample_equal(space1.sample(), space2.sample())


@pytest.mark.parametrize(
    "space",
    [
        Tuple([Discrete(100), Discrete(100)]),
        Tuple([Discrete(5), Discrete(10)]),
        Tuple(
            [
                Discrete(5),
                Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32),
            ]
        ),
        Tuple((Discrete(5), Discrete(2), Discrete(2))),
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
def test_seed_subspace_incorrelated(space):
    subspaces = space.spaces if isinstance(space, Tuple) else space.spaces.values()

    space.seed(0)
    states = [
        convert_sample_hashable(subspace.np_random.get_state())
        for subspace in subspaces
    ]

    assert len(states) == len(set(states))


def test_multidiscrete_as_tuple():
    # 1D multi-discrete
    space = MultiDiscrete([3, 4, 5])

    assert space.shape == (3,)
    assert space[0] == Discrete(3)
    assert space[0:1] == MultiDiscrete([3])
    assert space[0:2] == MultiDiscrete([3, 4])
    assert space[:] == space and space[:] is not space
    assert len(space) == 3

    # 2D multi-discrete
    space = MultiDiscrete([[3, 4, 5], [6, 7, 8]])

    assert space.shape == (2, 3)
    assert space[0, 1] == Discrete(4)
    assert space[0] == MultiDiscrete([3, 4, 5])
    assert space[0:1] == MultiDiscrete([[3, 4, 5]])
    assert space[0:2, :] == MultiDiscrete([[3, 4, 5], [6, 7, 8]])
    assert space[:, 0:1] == MultiDiscrete([[3], [6]])
    assert space[0:2, 0:2] == MultiDiscrete([[3, 4], [6, 7]])
    assert space[:] == space and space[:] is not space
    assert space[:, :] == space and space[:, :] is not space


def test_multidiscrete_subspace_reproducibility():
    # 1D multi-discrete
    space = MultiDiscrete([100, 200, 300])
    space.seed(None)

    assert sample_equal(space[0].sample(), space[0].sample())
    assert sample_equal(space[0:1].sample(), space[0:1].sample())
    assert sample_equal(space[0:2].sample(), space[0:2].sample())
    assert sample_equal(space[:].sample(), space[:].sample())
    assert sample_equal(space[:].sample(), space.sample())

    # 2D multi-discrete
    space = MultiDiscrete([[300, 400, 500], [600, 700, 800]])
    space.seed(None)

    assert sample_equal(space[0, 1].sample(), space[0, 1].sample())
    assert sample_equal(space[0].sample(), space[0].sample())
    assert sample_equal(space[0:1].sample(), space[0:1].sample())
    assert sample_equal(space[0:2, :].sample(), space[0:2, :].sample())
    assert sample_equal(space[:, 0:1].sample(), space[:, 0:1].sample())
    assert sample_equal(space[0:2, 0:2].sample(), space[0:2, 0:2].sample())
    assert sample_equal(space[:].sample(), space[:].sample())
    assert sample_equal(space[:, :].sample(), space[:, :].sample())
    assert sample_equal(space[:, :].sample(), space.sample())


def test_space_legacy_state_pickling():
    legacy_state = {
        "shape": (
            1,
            2,
            3,
        ),
        "dtype": np.int64,
        "np_random": np.random.default_rng(),
        "n": 3,
    }
    space = Discrete(1)
    space.__setstate__(legacy_state)

    assert space.shape == legacy_state["shape"]
    assert space._shape == legacy_state["shape"]
    assert space.np_random == legacy_state["np_random"]
    assert space._np_random == legacy_state["np_random"]
    assert space.n == 3
    assert space.dtype == legacy_state["dtype"]
