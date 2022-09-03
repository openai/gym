import copy
import itertools
import json  # note: ujson fails this test due to float equality
import pickle
import tempfile
from typing import List, Union

import numpy as np
import pytest

from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space, Text
from gym.utils import seeding
from gym.utils.env_checker import data_equivalence
from tests.spaces.utils import (
    TESTING_FUNDAMENTAL_SPACES,
    TESTING_FUNDAMENTAL_SPACES_IDS,
    TESTING_SPACES,
    TESTING_SPACES_IDS,
)

# Due to this test taking a 1ms each then we don't mind generating so many tests
# This generates all pairs of spaces of the same type in TESTING_SPACES
TESTING_SPACES_PERMUTATIONS = list(
    itertools.chain(
        *[
            list(itertools.permutations(list(group), r=2))
            for key, group in itertools.groupby(
                TESTING_SPACES, key=lambda space: type(space)
            )
        ]
    )
)


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_roundtripping(space: Space):
    """Tests if space samples passed to `to_jsonable` and `from_jsonable` produce the original samples."""
    sample_1 = space.sample()
    sample_2 = space.sample()

    # Convert the samples to json, dump + load json and convert back to python
    sample_json = space.to_jsonable([sample_1, sample_2])
    sample_roundtripped = json.loads(json.dumps(sample_json))
    sample_1_prime, sample_2_prime = space.from_jsonable(sample_roundtripped)

    # Check if the samples are equivalent
    assert data_equivalence(
        sample_1, sample_1_prime
    ), f"sample 1: {sample_1}, prime: {sample_1_prime}"
    assert data_equivalence(
        sample_2, sample_2_prime
    ), f"sample 2: {sample_2}, prime: {sample_2_prime}"


@pytest.mark.parametrize(
    "space_1,space_2",
    TESTING_SPACES_PERMUTATIONS,
    ids=[f"({s1}, {s2})" for s1, s2 in TESTING_SPACES_PERMUTATIONS],
)
def test_space_equality(space_1, space_2):
    """Check that `space.__eq__` works.

    Testing spaces permutations contains all combinations of testing spaces of the same type.
    """
    assert space_1 == space_1
    assert space_2 == space_2
    assert space_1 != space_2


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
    "space", TESTING_FUNDAMENTAL_SPACES, ids=TESTING_FUNDAMENTAL_SPACES_IDS
)
def test_sample(space: Space, n_trials: int = 1_000):
    """Test the space sample has the expected distribution with the chi-squared test and KS test.

    Example code with scipy.stats.chisquared that should have the same

    >>> import scipy.stats
    >>> variance = np.sum(np.square(observed_frequency - expected_frequency) / expected_frequency)
    >>> f'X2 at alpha=0.05 = {scipy.stats.chi2.isf(0.05, df=4)}'
    >>> f'p-value = {scipy.stats.chi2.sf(variance, df=4)}'
    >>> scipy.stats.chisquare(f_obs=observed_frequency)
    """
    space.seed(0)
    samples = np.array([space.sample() for _ in range(n_trials)])
    assert len(samples) == n_trials

    if isinstance(space, Box):
        # TODO: Add KS testing for continuous uniform distribution
        pass
    elif isinstance(space, Discrete):
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
    elif isinstance(space, Text):
        expected_frequency = (
            np.ones(len(space.character_set))
            * n_trials
            * (space.min_length + (space.max_length - space.min_length) / 2)
            / len(space.character_set)
        )
        observed_frequency = np.zeros(len(space.character_set))
        for sample in samples:
            for x in sample:
                observed_frequency[space.character_index(x)] += 1
        degrees_of_freedom = len(space.character_set) - 1

        assert observed_frequency.shape == expected_frequency.shape
        assert np.sum(observed_frequency) == sum(len(sample) for sample in samples)

        variance = np.sum(
            np.square(expected_frequency - observed_frequency) / expected_frequency
        )
        if degrees_of_freedom == 61:
            # scipy.stats.chi2.isf(0.05, df=61)
            assert variance < 80.23209784876272
        else:
            assert variance < CHI_SQUARED[degrees_of_freedom]
    else:
        raise NotImplementedError(f"Unknown sample testing for {type(space)}")


SAMPLE_MASK_RNG, _ = seeding.np_random(1)


@pytest.mark.parametrize(
    "space,mask",
    itertools.zip_longest(
        TESTING_FUNDAMENTAL_SPACES,
        [
            # Discrete
            np.array([1, 1, 0], dtype=np.int8),
            np.array([0, 0, 0], dtype=np.int8),
            # Box
            None,
            None,
            None,
            None,
            None,
            # Multi-discrete
            (np.array([1, 1], dtype=np.int8), np.array([0, 0], dtype=np.int8)),
            (
                (np.array([1, 0], dtype=np.int8), np.array([0, 1, 1], dtype=np.int8)),
                (np.array([1, 1, 0], dtype=np.int8), np.array([0, 1], dtype=np.int8)),
            ),
            # Multi-binary
            np.array([0, 1, 0, 1, 0, 2, 1, 1], dtype=np.int8),
            np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int8),
            # Text
            (None, SAMPLE_MASK_RNG.integers(low=0, high=2, size=62, dtype=np.int8)),
            (4, SAMPLE_MASK_RNG.integers(low=0, high=2, size=62, dtype=np.int8)),
            (None, np.array([1, 1, 0, 1, 0, 0], dtype=np.int8)),
        ],
    ),
    ids=TESTING_FUNDAMENTAL_SPACES_IDS,
)
def test_space_sample_mask(space: Space, mask, n_trials: int = 100):
    """Tests that the sampling a space with a mask has the expected distribution.

    The implemented code is similar to the `test_space_sample` that considers the mask applied.
    """
    if isinstance(space, Box):
        # The box space can't have a sample mask
        assert mask is None
        return
    assert mask is not None

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
        expected_frequency = (
            np.ones(space.shape) * np.where(mask == 2, 0.5, mask) * n_trials
        )
        print(expected_frequency)
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
    elif isinstance(space, Text):
        length, charlist_mask = mask

        if length is None:
            expected_length = (
                space.min_length + (space.max_length - space.min_length) / 2
            )
        else:
            expected_length = length

        if np.any(charlist_mask == 1):
            expected_frequency = (
                np.ones(len(space.character_set))
                * n_trials
                * expected_length
                / np.sum(charlist_mask)
                * charlist_mask
            )
        else:
            expected_frequency = np.zeros(len(space.character_set))

        observed_frequency = np.zeros(len(space.character_set))
        for sample in samples:
            for char in sample:
                observed_frequency[space.character_index(char)] += 1

        degrees_of_freedom = max(np.sum(charlist_mask) - 1, 0)

        assert observed_frequency.shape == expected_frequency.shape
        assert np.sum(observed_frequency) == sum(len(sample) for sample in samples)

        variance = np.sum(
            np.square(expected_frequency - observed_frequency)
            / np.clip(expected_frequency, 1, None)
        )
        if degrees_of_freedom == 26:
            # scipy.stats.chi2.isf(0.05, df=29)
            assert variance < 38.88513865983007
        elif degrees_of_freedom == 31:
            # scipy.stats.chi2.isf(0.05, df=31)
            assert variance < 44.985343280365136
        else:
            assert variance < CHI_SQUARED[degrees_of_freedom]
    else:
        raise NotImplementedError()


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_seed_reproducibility(space):
    """Test that the set the space seed will reproduce the same samples."""
    space_1 = space
    space_2 = copy.deepcopy(space)

    for seed in range(5):
        assert space_1.seed(seed) == space_2.seed(seed)
        # With the same seed, the two spaces should be identical
        assert all(
            data_equivalence(space_1.sample(), space_2.sample()) for _ in range(10)
        )

    assert space_1.seed(123) != space_2.seed(456)
    # Due to randomness, it is difficult to test that random seeds produce different answers
    #   Therefore, taking 10 samples and checking that they are not all the same.
    assert not all(
        data_equivalence(space_1.sample(), space_2.sample()) for _ in range(10)
    )


SPACE_CLS = list(dict.fromkeys(type(space) for space in TESTING_SPACES))
SPACE_KWARGS = [
    {"n": 3},  # Discrete
    {"low": 1, "high": 10},  # Box
    {"nvec": [3, 2]},  # MultiDiscrete
    {"n": 2},  # MultiBinary
    {"max_length": 5},  # Text
    {"spaces": (Discrete(3), Discrete(2))},  # Tuple
    {"spaces": {"a": Discrete(3), "b": Discrete(2)}},  # Dict
    {"node_space": Discrete(4), "edge_space": Discrete(3)},  # Graph
    {"space": Discrete(4)},  # Sequence
]
assert len(SPACE_CLS) == len(SPACE_KWARGS)


@pytest.mark.parametrize(
    "space_cls,kwarg",
    list(zip(SPACE_CLS, SPACE_KWARGS)),
    ids=[f"{space_cls}" for space_cls in SPACE_CLS],
)
def test_seed_np_random(space_cls, kwarg):
    """During initialisation of a space, a rng instance can be passed to the space.

    Test that the space's `np_random` is the rng instance
    """
    rng, _ = seeding.np_random(123)

    space = space_cls(seed=rng, **kwarg)
    assert space.np_random is rng


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_sample_contains(space):
    """Test that samples are contained within the space.

    Then test that for all other spaces, we test that an error is not raise with a sample and a bool is returned.
    As other spaces can be contained with this space, we cannot test that the contains is always true or false.
    """
    for _ in range(10):
        sample = space.sample()
        assert sample in space
        assert space.contains(sample)

    for other_space in TESTING_SPACES:
        assert isinstance(space.contains(other_space.sample()), bool)


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_repr(space):
    assert isinstance(str(space), str)


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_space_pickling(space):
    """Tests the spaces can be pickled with the unpickled version being equivalent to the original."""
    space.seed(0)

    # Pickle and unpickle with a string
    pickled_space = pickle.dumps(space)
    unpickled_space = pickle.loads(pickled_space)
    assert space == unpickled_space

    # Pickle and unpickle with a file
    with tempfile.TemporaryFile() as f:
        pickle.dump(space, f)
        f.seek(0)
        file_unpickled_space = pickle.load(f)

    assert space == file_unpickled_space

    # Check that space samples are the same
    space_sample = space.sample()
    unpickled_sample = unpickled_space.sample()
    file_unpickled_sample = file_unpickled_space.sample()
    assert data_equivalence(space_sample, unpickled_sample)
    assert data_equivalence(space_sample, file_unpickled_sample)
