import copy
import json  # note: ujson fails this test due to float equality
import pickle
import string
import tempfile
from typing import List, Union

import numpy as np
import pytest

from gym.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Space,
    Text,
    Tuple,
)


@pytest.mark.parametrize(
    "space",
    [
        Discrete(3),
        Discrete(5, start=-2),
        Box(low=0.0, high=np.inf, shape=(2, 2)),
        Tuple([Discrete(5), Discrete(10)]),
        Tuple(
            [
                Discrete(5),
                Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 5.0]),
                    dtype=np.float64,
                ),
            ]
        ),
        Tuple((Discrete(5), Discrete(2), Discrete(2))),
        Tuple((Discrete(5), Discrete(2, start=6), Discrete(2, start=-4))),
        MultiDiscrete([2, 2, 100]),
        MultiBinary(10),
        Dict(
            {
                "position": Discrete(5),
                "velocity": Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 5.0]),
                    dtype=np.float64,
                ),
            }
        ),
        Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5)),
        Graph(node_space=Discrete(5), edge_space=Box(low=-100, high=100, shape=(3, 4))),
        Graph(node_space=Discrete(5), edge_space=None),
        Sequence(Discrete(4)),
        Sequence(Dict({"feature": Box(0, 1, (3,))})),
        Text(5),
        Text(min_length=1, max_length=10, charset=string.digits),
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
    assert s1 == s1p, f"Expected {s1} to equal {s1p}"
    assert s2 == s2p, f"Expected {s2} to equal {s2p}"


@pytest.mark.parametrize(
    "space",
    [
        Discrete(3),
        Discrete(5, start=-2),
        Box(low=np.array([-10.0, 0.0]), high=np.array([10.0, 10.0]), dtype=np.float64),
        Box(low=-np.inf, high=np.inf, shape=(1, 3)),
        Tuple([Discrete(5), Discrete(10)]),
        Tuple(
            [
                Discrete(5),
                Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 5.0]),
                    dtype=np.float64,
                ),
            ]
        ),
        Tuple((Discrete(5), Discrete(2), Discrete(2))),
        Tuple((Discrete(5), Discrete(2), Discrete(2, start=-6))),
        MultiDiscrete([2, 2, 100]),
        MultiBinary(6),
        Dict(
            {
                "position": Discrete(5),
                "velocity": Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 5.0]),
                    dtype=np.float64,
                ),
            }
        ),
        Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5)),
        Graph(node_space=Discrete(5), edge_space=Box(low=-100, high=100, shape=(3, 4))),
        Graph(node_space=Discrete(5), edge_space=None),
        Sequence(Discrete(4)),
        Sequence(Dict({"feature": Box(0, 1, (3,))})),
        Text(5),
        Text(min_length=1, max_length=10, charset=string.digits),
    ],
)
def test_equality(space):
    space1 = space
    space2 = copy.deepcopy(space)
    assert space1 == space2, f"Expected {space1} to equal {space2}"


@pytest.mark.parametrize(
    "spaces",
    [
        (Discrete(3), Discrete(4)),
        (Discrete(3), Discrete(3, start=-1)),
        (MultiDiscrete([2, 2, 100]), MultiDiscrete([2, 2, 8])),
        (MultiBinary(8), MultiBinary(7)),
        (
            Box(
                low=np.array([-10.0, 0.0]),
                high=np.array([10.0, 10.0]),
                dtype=np.float64,
            ),
            Box(
                low=np.array([-10.0, 0.0]), high=np.array([10.0, 9.0]), dtype=np.float64
            ),
        ),
        (
            Box(low=-np.inf, high=0.0, shape=(2, 1)),
            Box(low=0.0, high=np.inf, shape=(2, 1)),
        ),
        (Tuple([Discrete(5), Discrete(10)]), Tuple([Discrete(1), Discrete(10)])),
        (
            Tuple([Discrete(5), Discrete(10)]),
            Tuple([Discrete(5, start=7), Discrete(10)]),
        ),
        (Dict({"position": Discrete(5)}), Dict({"position": Discrete(4)})),
        (Dict({"position": Discrete(5)}), Dict({"speed": Discrete(5)})),
        (
            Graph(
                node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5)
            ),
            Graph(node_space=Discrete(5), edge_space=None),
        ),
        (
            Sequence(Discrete(4)),
            Sequence(Dict({"feature": Box(0, 1, (3,))})),
        ),
        (Sequence(Discrete(4)), Sequence(Discrete(4, start=-1))),
        (
            Text(5),
            Text(min_length=1, max_length=10, charset=string.digits),
        ),
    ],
)
def test_inequality(spaces):
    space1, space2 = spaces
    assert space1 != space2, f"Expected {space1} != {space2}"


# The expected sum of variance for an alpha of 0.05
# CHI_SQUARED = [0] + [scipy.stats.chi2.isf(0.05, df=df) for df in range(1, 25)]
CHI_SQUARED = np.array(
    [
        0.01,
        3.8414588206941285,
        5.991464547107983,
        7.814727903251178,
        9.487729036781158,
        11.070497693516355,
        12.59158724374398,
        14.067140449340167,
        15.507313055865454,
        16.91897760462045,
    ]
)


@pytest.mark.parametrize(
    "space",
    [
        Discrete(1),
        Discrete(5),
        Discrete(8, start=-20),
        Box(low=0, high=255, shape=(2,), dtype=np.uint8),
        Box(low=-np.inf, high=np.inf, shape=(3,)),
        Box(low=1.0, high=np.inf, shape=(3,)),
        Box(low=-np.inf, high=2.0, shape=(3,)),
        Box(low=np.array([0, 2]), high=np.array([10, 4])),
        MultiDiscrete([3, 5]),
        MultiDiscrete(np.array([[3, 5], [2, 1]])),
        MultiBinary([2, 4]),
    ],
)
def test_sample(space: Space, n_trials: int = 1_000):
    """Test the space sample has the expected distribution with the chi-squared test and KS test.

    Example code with scipy.stats.chisquared

    import scipy.stats
    variance = np.sum(np.square(observed_frequency - expected_frequency) / expected_frequency)
    f'X2 at alpha=0.05 = {scipy.stats.chi2.isf(0.05, df=4)}'
    f'p-value = {scipy.stats.chi2.sf(variance, df=4)}'
    scipy.stats.chisquare(f_obs=observed_frequency)
    """
    space.seed(0)
    samples = np.array([space.sample() for _ in range(n_trials)])
    assert len(samples) == n_trials

    # todo add Box space test
    if isinstance(space, Discrete):
        expected_frequency = np.ones(space.n) * n_trials / space.n
        observed_frequency = np.zeros(space.n)
        for sample in samples:
            observed_frequency[sample - space.start] += 1
        degrees_of_freedom = space.n - 1

        assert observed_frequency.shape == expected_frequency.shape
        assert np.sum(observed_frequency) == n_trials

        variance = np.sum(
            np.square(expected_frequency - observed_frequency) / expected_frequency
        )
        assert variance < CHI_SQUARED[degrees_of_freedom]
    elif isinstance(space, MultiBinary):
        expected_frequency = n_trials / 2
        observed_frequency = np.sum(samples, axis=0)
        assert observed_frequency.shape == space.shape

        # As this is a binary space, then we can be lazy in the variance as the np.square is symmetric for the 0 and 1 categories
        variance = (
            2 * np.square(observed_frequency - expected_frequency) / expected_frequency
        )
        assert variance.shape == space.shape
        assert np.all(variance < CHI_SQUARED[1])
    elif isinstance(space, MultiDiscrete):
        # Due to the multi-axis capability of MultiDiscrete, these functions need to be recursive and that the expected / observed numpy are of non-regular shapes
        def _generate_frequency(dim, func):
            if isinstance(dim, np.ndarray):
                return np.array(
                    [_generate_frequency(sub_dim, func) for sub_dim in dim],
                    dtype=object,
                )
            else:
                return func(dim)

        def _update_observed_frequency(obs_sample, obs_freq):
            if isinstance(obs_sample, np.ndarray):
                for sub_sample, sub_freq in zip(obs_sample, obs_freq):
                    _update_observed_frequency(sub_sample, sub_freq)
            else:
                obs_freq[obs_sample] += 1

        expected_frequency = _generate_frequency(
            space.nvec, lambda dim: np.ones(dim) * n_trials / dim
        )
        observed_frequency = _generate_frequency(space.nvec, lambda dim: np.zeros(dim))
        for sample in samples:
            _update_observed_frequency(sample, observed_frequency)

        def _chi_squared_test(dim, exp_freq, obs_freq):
            if isinstance(dim, np.ndarray):
                for sub_dim, sub_exp_freq, sub_obs_freq in zip(dim, exp_freq, obs_freq):
                    _chi_squared_test(sub_dim, sub_exp_freq, sub_obs_freq)
            else:
                assert exp_freq.shape == (dim,) and obs_freq.shape == (dim,)
                assert np.sum(obs_freq) == n_trials
                assert np.sum(exp_freq) == n_trials
                _variance = np.sum(np.square(exp_freq - obs_freq) / exp_freq)
                _degrees_of_freedom = dim - 1
                assert _variance < CHI_SQUARED[_degrees_of_freedom]

        _chi_squared_test(space.nvec, expected_frequency, observed_frequency)


@pytest.mark.parametrize(
    "space,mask",
    [
        (Discrete(5), np.array([0, 1, 1, 0, 1], dtype=np.int8)),
        (Discrete(4, start=-20), np.array([1, 1, 0, 1], dtype=np.int8)),
        (Discrete(4, start=1), np.array([0, 0, 0, 0], dtype=np.int8)),
        (MultiBinary([3, 2]), np.array([[0, 1], [1, 1], [0, 0]], dtype=np.int8)),
        (
            MultiDiscrete([5, 3]),
            (
                np.array([0, 1, 1, 0, 1], dtype=np.int8),
                np.array([0, 1, 1], dtype=np.int8),
            ),
        ),
        (
            MultiDiscrete(np.array([4, 2])),
            (np.array([0, 0, 0, 0], dtype=np.int8), np.array([1, 1], dtype=np.int8)),
        ),
        (
            MultiDiscrete(np.array([[2, 2], [4, 3]])),
            (
                (np.array([0, 1], dtype=np.int8), np.array([1, 1], dtype=np.int8)),
                (
                    np.array([0, 1, 1, 0], dtype=np.int8),
                    np.array([1, 0, 0], dtype=np.int8),
                ),
            ),
        ),
    ],
)
def test_space_sample_mask(space, mask, n_trials: int = 100):
    """Test the space sample with mask works using the pearson chi-squared test."""
    space.seed(1)
    samples = np.array([space.sample(mask) for _ in range(n_trials)])

    if isinstance(space, Discrete):
        if np.any(mask == 1):
            expected_frequency = np.ones(space.n) * (n_trials / np.sum(mask)) * mask
        else:
            expected_frequency = np.zeros(space.n)
            expected_frequency[0] = n_trials
        observed_frequency = np.zeros(space.n)
        for sample in samples:
            observed_frequency[sample - space.start] += 1
        degrees_of_freedom = max(np.sum(mask) - 1, 0)

        assert observed_frequency.shape == expected_frequency.shape
        assert np.sum(observed_frequency) == n_trials
        assert np.sum(expected_frequency) == n_trials
        variance = np.sum(
            np.square(expected_frequency - observed_frequency)
            / np.clip(expected_frequency, 1, None)
        )
        assert variance < CHI_SQUARED[degrees_of_freedom]
    elif isinstance(space, MultiBinary):
        expected_frequency = np.ones(space.shape) * mask * (n_trials / 2)
        observed_frequency = np.sum(samples, axis=0)
        assert space.shape == expected_frequency.shape == observed_frequency.shape

        variance = (
            2
            * np.square(observed_frequency - expected_frequency)
            / np.clip(expected_frequency, 1, None)
        )
        assert variance.shape == space.shape
        assert np.all(variance < CHI_SQUARED[1])
    elif isinstance(space, MultiDiscrete):
        # Due to the multi-axis capability of MultiDiscrete, these functions need to be recursive and that the expected / observed numpy are of non-regular shapes
        def _generate_frequency(
            _dim: Union[np.ndarray, int], _mask, func: callable
        ) -> List:
            if isinstance(_dim, np.ndarray):
                return [
                    _generate_frequency(sub_dim, sub_mask, func)
                    for sub_dim, sub_mask in zip(_dim, _mask)
                ]
            else:
                return func(_dim, _mask)

        def _update_observed_frequency(obs_sample, obs_freq):
            if isinstance(obs_sample, np.ndarray):
                for sub_sample, sub_freq in zip(obs_sample, obs_freq):
                    _update_observed_frequency(sub_sample, sub_freq)
            else:
                obs_freq[obs_sample] += 1

        def _exp_freq_fn(_dim: int, _mask: np.ndarray):
            if np.any(_mask == 1):
                assert _dim == len(_mask)
                return np.ones(_dim) * (n_trials / np.sum(_mask)) * _mask
            else:
                freq = np.zeros(_dim)
                freq[0] = n_trials
                return freq

        expected_frequency = _generate_frequency(
            space.nvec, mask, lambda dim, _mask: _exp_freq_fn(dim, _mask)
        )
        observed_frequency = _generate_frequency(
            space.nvec, mask, lambda dim, _: np.zeros(dim)
        )
        for sample in samples:
            _update_observed_frequency(sample, observed_frequency)

        def _chi_squared_test(dim, _mask, exp_freq, obs_freq):
            if isinstance(dim, np.ndarray):
                for sub_dim, sub_mask, sub_exp_freq, sub_obs_freq in zip(
                    dim, _mask, exp_freq, obs_freq
                ):
                    _chi_squared_test(sub_dim, sub_mask, sub_exp_freq, sub_obs_freq)
            else:
                assert exp_freq.shape == (dim,) and obs_freq.shape == (dim,)
                assert np.sum(obs_freq) == n_trials
                assert np.sum(exp_freq) == n_trials
                _variance = np.sum(
                    np.square(exp_freq - obs_freq) / np.clip(exp_freq, 1, None)
                )
                _degrees_of_freedom = max(np.sum(_mask) - 1, 0)
                assert _variance < CHI_SQUARED[_degrees_of_freedom]

        _chi_squared_test(space.nvec, mask, expected_frequency, observed_frequency)
    else:
        raise NotImplementedError()


@pytest.mark.parametrize(
    "space,mask",
    [
        (
            Dict(a=Discrete(2), b=MultiDiscrete([2, 4])),
            {
                "a": np.array([0, 1], dtype=np.int8),
                "b": (
                    np.array([0, 1], dtype=np.int8),
                    np.array([1, 1, 0, 0], dtype=np.int8),
                ),
            },
        ),
        (
            Tuple([Box(0, 1, ()), Discrete(3), MultiBinary([2, 1])]),
            (
                None,
                np.array([0, 1, 0], dtype=np.int8),
                np.array([[0], [1]], dtype=np.int8),
            ),
        ),
        (
            Dict(a=Tuple([Box(0, 1, ()), Discrete(3)]), b=Discrete(3)),
            {
                "a": (None, np.array([1, 0, 0], dtype=np.int8)),
                "b": np.array([0, 1, 1], dtype=np.int8),
            },
        ),
        (Graph(node_space=Discrete(5), edge_space=Discrete(3)), None),
        (
            Graph(node_space=Discrete(3), edge_space=Box(low=0, high=1, shape=(5,))),
            None,
        ),
        (
            Graph(
                node_space=Box(low=-100, high=100, shape=(3,)), edge_space=Discrete(3)
            ),
            None,
        ),
        (Sequence(Discrete(2)), (None, np.array([0, 1], dtype=np.int8))),
        (
            Sequence(Discrete(2)),
            (np.array([2, 3, 4], dtype=np.int8), np.array([0, 1], dtype=np.int8)),
        ),
        (Sequence(Discrete(2)), (np.array([2, 3, 4], dtype=np.int8), None)),
        (Sequence(Discrete(2)), (None, None)),
        (Sequence(Discrete(2)), None),
    ],
)
def test_composite_space_sample_mask(space, mask):
    """Test that composite space samples use the mask correctly."""
    space.sample(mask)


@pytest.mark.parametrize(
    "spaces",
    [
        (Discrete(5), MultiBinary(5)),
        (
            Box(
                low=np.array([-10.0, 0.0]),
                high=np.array([10.0, 10.0]),
                dtype=np.float64,
            ),
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
        (
            Graph(
                node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5)
            ),
            Graph(node_space=Discrete(5), edge_space=None),
        ),
        (Sequence(Discrete(4)), Sequence(Discrete(3))),
        (
            Text(5),
            Text(min_length=1, max_length=10, charset=string.digits),
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
    assert space.contains(np.array(0.5, dtype=np.float32))

    # float64 is not in float32 space
    assert not space.contains(np.array(0.5))
    assert not space.contains(np.array(1))


@pytest.mark.parametrize(
    "space",
    [
        Discrete(3),
        Discrete(3, start=-4),
        Box(low=0.0, high=np.inf, shape=(2, 2)),
        Tuple([Discrete(5), Discrete(10)]),
        Tuple(
            [
                Discrete(5),
                Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 5.0]),
                    dtype=np.float64,
                ),
            ]
        ),
        Tuple((Discrete(5), Discrete(2), Discrete(2))),
        MultiDiscrete([2, 2, 100]),
        MultiBinary(10),
        Dict(
            {
                "position": Discrete(5),
                "velocity": Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 5.0]),
                    dtype=np.float64,
                ),
            }
        ),
        Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5)),
        Graph(node_space=Discrete(5), edge_space=Box(low=-100, high=100, shape=(3, 4))),
        Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=None),
        Graph(node_space=Discrete(5), edge_space=None),
        Sequence(Discrete(4)),
        Sequence(Dict({"a": Box(0, 1), "b": Discrete(3)})),
        Text(5),
        Text(min_length=1, max_length=10, charset=string.digits),
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
        Discrete(3, start=-4),
        Box(low=0.0, high=np.inf, shape=(2, 2)),
        Tuple([Discrete(5), Discrete(10)]),
        Tuple(
            [
                Discrete(5),
                Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 5.0]),
                    dtype=np.float64,
                ),
            ]
        ),
        Tuple((Discrete(5), Discrete(2), Discrete(2))),
        MultiDiscrete([2, 2, 100]),
        MultiBinary(10),
        Dict(
            {
                "position": Discrete(5),
                "velocity": Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 5.0]),
                    dtype=np.float64,
                ),
            }
        ),
        Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5)),
        Graph(node_space=Discrete(5), edge_space=Box(low=-100, high=100, shape=(3, 4))),
        Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=None),
        Graph(node_space=Discrete(5), edge_space=None),
        Sequence(Discrete(4)),
        Sequence(Dict({"a": Box(0, 1), "b": Discrete(3)})),
        Text(5),
        Text(min_length=1, max_length=10, charset=string.digits),
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
        Tuple([Discrete(5), Discrete(5, start=10)]),
        Tuple(
            [
                Discrete(5),
                Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 5.0]),
                    dtype=np.float64,
                ),
            ]
        ),
        Tuple((Discrete(5), Discrete(2), Discrete(2))),
        Dict(
            {
                "position": Discrete(5),
                "velocity": Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 5.0]),
                    dtype=np.float64,
                ),
            }
        ),
        Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5)),
        Graph(node_space=Discrete(5), edge_space=Box(low=-100, high=100, shape=(3, 4))),
        Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=None),
        Graph(node_space=Discrete(5), edge_space=None),
        Text(5),
        Text(min_length=1, max_length=10, charset=string.digits),
    ],
)
def test_seed_subspace_incorrelated(space):
    subspaces = []
    if isinstance(space, Tuple):
        subspaces = space.spaces
    elif isinstance(space, Dict):
        subspaces = space.spaces.values()
    elif isinstance(space, Graph):
        if space.edge_space is not None:
            subspaces = [space.node_space, space.edge_space]
        else:
            subspaces = [space.node_space]

    space.seed(0)
    states = [
        convert_sample_hashable(subspace.np_random.bit_generator.state)
        for subspace in subspaces
    ]

    assert len(states) == len(set(states))


def test_tuple():
    spaces = [Discrete(5), Discrete(10), Discrete(5)]
    space_tuple = Tuple(spaces)

    assert len(space_tuple) == len(spaces)
    assert space_tuple.count(Discrete(5)) == 2
    assert space_tuple.count(MultiBinary(2)) == 0
    for i, space in enumerate(space_tuple):
        assert space == spaces[i]
    for i, space in enumerate(reversed(space_tuple)):
        assert space == spaces[len(spaces) - 1 - i]
    assert space_tuple.index(Discrete(5)) == 0
    assert space_tuple.index(Discrete(5), 1) == 2
    with pytest.raises(ValueError):
        space_tuple.index(Discrete(10), 0, 1)


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
    assert space._shape == legacy_state["shape"]  # pyright: reportPrivateUsage=false
    assert space.np_random == legacy_state["np_random"]
    assert (
        space._np_random == legacy_state["np_random"]
    )  # pyright: reportPrivateUsage=false
    assert space.n == 3
    assert space.dtype == legacy_state["dtype"]


@pytest.mark.parametrize(
    "space",
    [
        Box(low=0, high=np.inf, shape=(2,), dtype=np.int32),
        Box(low=0, high=np.inf, shape=(2,), dtype=np.float32),
        Box(low=0, high=np.inf, shape=(2,), dtype=np.int64),
        Box(low=0, high=np.inf, shape=(2,), dtype=np.float64),
        Box(low=-np.inf, high=0, shape=(2,), dtype=np.int32),
        Box(low=-np.inf, high=0, shape=(2,), dtype=np.float32),
        Box(low=-np.inf, high=0, shape=(2,), dtype=np.int64),
        Box(low=-np.inf, high=0, shape=(2,), dtype=np.float64),
        Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.int32),
        Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.int64),
        Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
        Box(low=0, high=np.inf, shape=(2, 3), dtype=np.int32),
        Box(low=0, high=np.inf, shape=(2, 3), dtype=np.float32),
        Box(low=0, high=np.inf, shape=(2, 3), dtype=np.int64),
        Box(low=0, high=np.inf, shape=(2, 3), dtype=np.float64),
        Box(low=-np.inf, high=0, shape=(2, 3), dtype=np.int32),
        Box(low=-np.inf, high=0, shape=(2, 3), dtype=np.float32),
        Box(low=-np.inf, high=0, shape=(2, 3), dtype=np.int64),
        Box(low=-np.inf, high=0, shape=(2, 3), dtype=np.float64),
        Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.int32),
        Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.float32),
        Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.int64),
        Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.float64),
        Box(low=np.array([-np.inf, 0]), high=np.array([0.0, np.inf]), dtype=np.int32),
        Box(low=np.array([-np.inf, 0]), high=np.array([0.0, np.inf]), dtype=np.float32),
        Box(low=np.array([-np.inf, 0]), high=np.array([0.0, np.inf]), dtype=np.int64),
        Box(low=np.array([-np.inf, 0]), high=np.array([0.0, np.inf]), dtype=np.float64),
    ],
)
def test_infinite_space(space):
    # for this test, make sure that spaces that are passed in have only 0 or infinite bounds
    # because space.high and space.low are both modified within the init
    # so we check for infinite when we know it's not 0
    space.seed(0)

    assert np.all(space.high > space.low), "High bound not higher than low bound"

    sample = space.sample()

    # check if space contains sample
    assert space.contains(
        sample
    ), "Sample {sample} not inside space according to `space.contains()`"

    # manually check that the sign of the sample is within the bounds
    assert np.all(
        np.sign(space.high) >= np.sign(sample)
    ), f"Sign of sample {sample} is less than space upper bound {space.high}"
    assert np.all(
        np.sign(space.low) <= np.sign(sample)
    ), f"Sign of sample {sample} is more than space lower bound {space.low}"

    # check that int bounds are bounded for everything
    # but floats are unbounded for infinite
    if np.any(space.high != 0):
        assert (
            space.is_bounded("above") is False
        ), "inf upper bound supposed to be unbounded"
    else:
        assert (
            space.is_bounded("above") is True
        ), "non-inf upper bound supposed to be bounded"

    if np.any(space.low != 0):
        assert (
            space.is_bounded("below") is False
        ), "inf lower bound supposed to be unbounded"
    else:
        assert (
            space.is_bounded("below") is True
        ), "non-inf lower bound supposed to be bounded"

    # check for dtype
    assert (
        space.high.dtype == space.dtype
    ), "High's dtype {space.high.dtype} doesn't match `space.dtype`'"
    assert (
        space.low.dtype == space.dtype
    ), "Low's dtype {space.high.dtype} doesn't match `space.dtype`'"


def test_discrete_legacy_state_pickling():
    legacy_state = {
        "n": 3,
    }

    d = Discrete(1)
    assert "start" in d.__dict__
    del d.__dict__["start"]  # legacy did not include start param
    assert "start" not in d.__dict__

    d.__setstate__(legacy_state)

    assert d.start == 0
    assert d.n == 3


def test_box_legacy_state_pickling():
    legacy_state = {
        "dtype": np.dtype("float32"),
        "_shape": (5,),
        "low": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "high": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "bounded_below": np.array([True, True, True, True, True]),
        "bounded_above": np.array([True, True, True, True, True]),
        "_np_random": None,
    }

    b = Box(-1, 1, ())
    assert "low_repr" in b.__dict__ and "high_repr" in b.__dict__
    del b.__dict__["low_repr"]
    del b.__dict__["high_repr"]
    assert "low_repr" not in b.__dict__ and "high_repr" not in b.__dict__

    b.__setstate__(legacy_state)
    assert b.low_repr == "0.0"
    assert b.high_repr == "1.0"


@pytest.mark.parametrize(
    "space",
    [
        Discrete(3),
        Discrete(5, start=-2),
        Box(low=0.0, high=np.inf, shape=(2, 2)),
        Tuple([Discrete(5), Discrete(10)]),
        Tuple(
            [
                Discrete(5),
                Box(low=np.array([0.0, 0.0]), high=np.array([1, 5]), dtype=np.float64),
            ]
        ),
        Tuple((Discrete(5), Discrete(2), Discrete(2))),
        Tuple((Discrete(5), Discrete(2, start=6), Discrete(2, start=-4))),
        MultiDiscrete([2, 2, 100]),
        MultiBinary(10),
        Dict(
            {
                "position": Discrete(5),
                "velocity": Box(
                    low=np.array([0.0, 0.0]), high=np.array([1, 5]), dtype=np.float64
                ),
            }
        ),
        Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5)),
        Graph(node_space=Discrete(5), edge_space=Box(low=-100, high=100, shape=(3, 4))),
        Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=None),
        Graph(node_space=Discrete(5), edge_space=None),
        Sequence(Discrete(4)),
        Sequence(Dict({"a": Box(0, 1), "b": Discrete(3)})),
        Text(5),
        Text(min_length=1, max_length=10, charset=string.digits),
    ],
)
def test_pickle(space):
    space.sample()

    # Pickle and unpickle with a string
    pickled = pickle.dumps(space)
    space2 = pickle.loads(pickled)

    # Pickle and unpickle with a file
    with tempfile.TemporaryFile() as f:
        pickle.dump(space, f)
        f.seek(0)
        space3 = pickle.load(f)

    sample = space.sample()
    sample2 = space2.sample()
    sample3 = space3.sample()
    assert sample_equal(sample, sample2)
    assert sample_equal(sample, sample3)
