import re
from collections import OrderedDict

import numpy as np
import pytest

from gym.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    GraphInstance,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Tuple,
    utils,
)

homogeneous_spaces = [
    Discrete(3),
    Box(low=0.0, high=np.inf, shape=(2, 2)),
    Box(low=0.0, high=np.inf, shape=(2, 2), dtype=np.float16),
    Tuple([Discrete(5), Discrete(10)]),
    Tuple(
        [
            Discrete(5),
            Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 5.0]), dtype=np.float64),
        ]
    ),
    Tuple((Discrete(5), Discrete(2), Discrete(2))),
    MultiDiscrete([2, 2, 10]),
    MultiBinary(10),
    Dict(
        {
            "position": Discrete(5),
            "velocity": Box(
                low=np.array([0.0, 0.0]), high=np.array([1.0, 5.0]), dtype=np.float64
            ),
        }
    ),
    Discrete(3, start=2),
    Discrete(8, start=-5),
]

flatdims = [3, 4, 4, 15, 7, 9, 14, 10, 7, 3, 8]

non_homogenous_spaces = [
    Graph(node_space=Box(low=-100, high=100, shape=(2, 2)), edge_space=Discrete(5)),  #
    Graph(node_space=Discrete(5), edge_space=Box(low=-100, high=100, shape=(2, 2))),  #
    Graph(node_space=Discrete(5), edge_space=None),  #
    Sequence(Discrete(4)),  #
    Sequence(Box(-10, 10, shape=(2, 2))),  #
    Sequence(Tuple([Box(-10, 10, shape=(2,)), Box(-10, 10, shape=(2,))])),  #
    Dict(a=Sequence(Discrete(4)), b=Box(-10, 10, shape=(2, 2))),  #
    Dict(
        a=Graph(node_space=Discrete(4), edge_space=Discrete(4)),
        b=Box(-10, 10, shape=(2, 2)),
    ),  #
    Tuple([Sequence(Discrete(4)), Box(-10, 10, shape=(2, 2))]),  #
    Tuple(
        [
            Graph(node_space=Discrete(4), edge_space=Discrete(4)),
            Box(-10, 10, shape=(2, 2)),
        ]
    ),  #
    Sequence(Graph(node_space=Box(-100, 100, shape=(2, 2)), edge_space=Discrete(4))),  #
    Dict(
        a=Dict(
            a=Sequence(Box(-100, 100, shape=(2, 2))), b=Box(-100, 100, shape=(2, 2))
        ),
        b=Tuple([Box(-100, 100, shape=(2,)), Box(-100, 100, shape=(2,))]),
    ),  #
    Dict(
        a=Dict(
            a=Graph(node_space=Box(-100, 100, shape=(2, 2)), edge_space=None),
            b=Box(-100, 100, shape=(2, 2)),
        ),
        b=Tuple([Box(-100, 100, shape=(2,)), Box(-100, 100, shape=(2,))]),
    ),
]


@pytest.mark.parametrize("space", non_homogenous_spaces)
def test_non_flattenable(space):
    assert space.is_np_flattenable is False
    with pytest.raises(
        ValueError,
        match=re.escape(
            "cannot be flattened to a numpy array, probably because it contains a `Graph` or `Sequence` subspace"
        ),
    ):
        utils.flatdim(space)


@pytest.mark.parametrize(["space", "flatdim"], zip(homogeneous_spaces, flatdims))
def test_flatdim(space, flatdim):
    assert space.is_np_flattenable
    dim = utils.flatdim(space)
    assert dim == flatdim, f"Expected {dim} to equal {flatdim}"


@pytest.mark.parametrize("space", homogeneous_spaces)
def test_flatten_space_boxes(space):
    flat_space = utils.flatten_space(space)
    assert isinstance(flat_space, Box), f"Expected {type(flat_space)} to equal {Box}"
    flatdim = utils.flatdim(space)
    (single_dim,) = flat_space.shape
    assert single_dim == flatdim, f"Expected {single_dim} to equal {flatdim}"


@pytest.mark.parametrize("space", homogeneous_spaces + non_homogenous_spaces)
def test_flat_space_contains_flat_points(space):
    some_samples = [space.sample() for _ in range(10)]
    flattened_samples = [utils.flatten(space, sample) for sample in some_samples]
    flat_space = utils.flatten_space(space)
    for i, flat_sample in enumerate(flattened_samples):
        assert flat_space.contains(
            flat_sample
        ), f"Expected sample #{i} {flat_sample} to be in {flat_space}"


@pytest.mark.parametrize("space", homogeneous_spaces)
def test_flatten_dim(space):
    sample = utils.flatten(space, space.sample())
    (single_dim,) = sample.shape
    flatdim = utils.flatdim(space)
    assert single_dim == flatdim, f"Expected {single_dim} to equal {flatdim}"


@pytest.mark.parametrize("space", homogeneous_spaces + non_homogenous_spaces)
def test_flatten_roundtripping(space):
    some_samples = [space.sample() for _ in range(10)]
    flattened_samples = [utils.flatten(space, sample) for sample in some_samples]
    roundtripped_samples = [
        utils.unflatten(space, sample) for sample in flattened_samples
    ]
    for i, (original, roundtripped) in enumerate(
        zip(some_samples, roundtripped_samples)
    ):
        assert compare_nested(
            original, roundtripped
        ), f"Expected sample #{i} {original} to equal {roundtripped}"
        assert space.contains(roundtripped)


def compare_nested(left, right):
    if type(left) != type(right):
        return False
    elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.shape == right.shape and np.allclose(left, right)
    elif isinstance(left, OrderedDict) and isinstance(right, OrderedDict):
        res = len(left) == len(right)
        for ((left_key, left_value), (right_key, right_value)) in zip(
            left.items(), right.items()
        ):
            if not res:
                return False
            res = left_key == right_key and compare_nested(left_value, right_value)
        return res
    elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        res = len(left) == len(right)
        for (x, y) in zip(left, right):
            if not res:
                return False
            res = compare_nested(x, y)
        return res
    else:
        return left == right


"""
Expecteded flattened types are based off:
1. The type that the space is hardcoded as(ie. multi_discrete=np.int64, discrete=np.int64, multi_binary=np.int8)
2. The type that the space is instantiated with(ie. box=np.float32 by default unless instantiated with a different type)
3. The smallest type that the composite space(tuple, dict) can be represented as. In flatten, this is determined
   internally by numpy when np.concatenate is called.
"""

expected_flattened_dtypes = [
    np.int64,
    np.float32,
    np.float16,
    np.int64,
    np.float64,
    np.int64,
    np.int64,
    np.int8,
    np.float64,
    np.int64,
    np.int64,
]


@pytest.mark.parametrize(
    ["original_space", "expected_flattened_dtype"],
    zip(homogeneous_spaces, expected_flattened_dtypes),
)
def test_dtypes(original_space, expected_flattened_dtype):
    flattened_space = utils.flatten_space(original_space)

    original_sample = original_space.sample()
    flattened_sample = utils.flatten(original_space, original_sample)
    unflattened_sample = utils.unflatten(original_space, flattened_sample)

    assert flattened_space.contains(
        flattened_sample
    ), "Expected flattened_space to contain flattened_sample"
    assert (
        flattened_space.dtype == expected_flattened_dtype
    ), f"Expected flattened_space's dtype to equal {expected_flattened_dtype}"

    assert flattened_sample.dtype == flattened_space.dtype, (
        "Expected flattened_space's dtype to equal " "flattened_sample's dtype "
    )

    compare_sample_types(original_space, original_sample, unflattened_sample)


def compare_sample_types(original_space, original_sample, unflattened_sample):
    if isinstance(original_space, Discrete):
        assert isinstance(unflattened_sample, int), (
            "Expected unflattened_sample to be an int. unflattened_sample: "
            "{} original_sample: {}".format(unflattened_sample, original_sample)
        )
    elif isinstance(original_space, Tuple):
        for index in range(len(original_space)):
            compare_sample_types(
                original_space.spaces[index],
                original_sample[index],
                unflattened_sample[index],
            )
    elif isinstance(original_space, Dict):
        for key, space in original_space.spaces.items():
            compare_sample_types(space, original_sample[key], unflattened_sample[key])
    else:
        assert unflattened_sample.dtype == original_sample.dtype, (
            "Expected unflattened_sample's dtype to equal "
            "original_sample's dtype. unflattened_sample: "
            "{} original_sample: {}".format(unflattened_sample, original_sample)
        )


homogeneous_samples = [
    2,
    np.array([[1.0, 3.0], [5.0, 8.0]], dtype=np.float32),
    np.array([[1.0, 3.0], [5.0, 8.0]], dtype=np.float16),
    (3, 7),
    (2, np.array([0.5, 3.5], dtype=np.float32)),
    (3, 0, 1),
    np.array([0, 1, 7], dtype=np.int64),
    np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8),
    OrderedDict(
        [("position", 3), ("velocity", np.array([0.5, 3.5], dtype=np.float32))]
    ),
    3,
    -2,
]


expected_flattened_hom_samples = [
    np.array([0, 0, 1], dtype=np.int64),
    np.array([1.0, 3.0, 5.0, 8.0], dtype=np.float32),
    np.array([1.0, 3.0, 5.0, 8.0], dtype=np.float16),
    np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int64),
    np.array([0, 0, 1, 0, 0, 0.5, 3.5], dtype=np.float64),
    np.array([0, 0, 0, 1, 0, 1, 0, 0, 1], dtype=np.int64),
    np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int64),
    np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8),
    np.array([0, 0, 0, 1, 0, 0.5, 3.5], dtype=np.float64),
    np.array([0, 1, 0], dtype=np.int64),
    np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int64),
]

non_homogenous_samples = [
    GraphInstance(
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
        np.array(
            [
                0,
            ],
            dtype=int,
        ),
        np.array([[0, 1]], dtype=int),
    ),
    GraphInstance(
        np.array([0, 1], dtype=int),
        np.array([[[1, 2], [3, 4]]], dtype=np.float32),
        np.array([[0, 1]], dtype=int),
    ),
    GraphInstance(np.array([0, 1], dtype=int), None, np.array([[0, 1]], dtype=int)),
    (0, 1, 2),
    (
        np.array([[0, 1], [2, 3]], dtype=np.float32),
        np.array([[4, 5], [6, 7]], dtype=np.float32),
    ),
    (
        (np.array([0, 1], dtype=np.float32), np.array([2, 3], dtype=np.float32)),
        (np.array([4, 5], dtype=np.float32), np.array([6, 7], dtype=np.float32)),
    ),
    OrderedDict(
        [("a", (0, 1, 2)), ("b", np.array([[0, 1], [2, 3]], dtype=np.float32))]
    ),
    OrderedDict(
        [
            (
                "a",
                GraphInstance(
                    np.array([1, 2], dtype=np.int),
                    np.array(
                        [
                            0,
                        ],
                        dtype=int,
                    ),
                    np.array([[0, 1]], dtype=int),
                ),
            ),
            ("b", np.array([[0, 1], [2, 3]], dtype=np.float32)),
        ]
    ),
    ((0, 1, 2), np.array([[0, 1], [2, 3]], dtype=np.float32)),
    (
        GraphInstance(
            np.array([1, 2], dtype=np.int),
            np.array(
                [
                    0,
                ],
                dtype=int,
            ),
            np.array([[0, 1]], dtype=int),
        ),
        np.array([[0, 1], [2, 3]], dtype=np.float32),
    ),
    (
        GraphInstance(
            nodes=np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32),
            edges=np.array([0], dtype=int),
            edge_links=np.array([[0, 1]]),
        ),
        GraphInstance(
            nodes=np.array(
                [[[8, 9], [10, 11]], [[12, 13], [14, 15]]], dtype=np.float32
            ),
            edges=np.array([1], dtype=int),
            edge_links=np.array([[0, 1]]),
        ),
    ),
    OrderedDict(
        [
            (
                "a",
                OrderedDict(
                    [
                        (
                            "a",
                            (
                                np.array([[0, 1], [2, 3]], dtype=np.float32),
                                np.array([[4, 5], [6, 7]], dtype=np.float32),
                            ),
                        ),
                        ("b", np.array([[8, 9], [10, 11]], dtype=np.float32)),
                    ]
                ),
            ),
            (
                "b",
                (
                    np.array([12, 13], dtype=np.float32),
                    np.array([14, 15], dtype=np.float32),
                ),
            ),
        ]
    ),
    OrderedDict(
        [
            (
                "a",
                OrderedDict(
                    [
                        (
                            "a",
                            GraphInstance(
                                np.array(
                                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                                    dtype=np.float32,
                                ),
                                None,
                                np.array([[0, 1]], dtype=int),
                            ),
                        ),
                        ("b", np.array([[8, 9], [10, 11]], dtype=np.float32)),
                    ]
                ),
            ),
            (
                "b",
                (
                    np.array([12, 13], dtype=np.float32),
                    np.array([14, 15], dtype=np.float32),
                ),
            ),
        ]
    ),
]


expected_flattened_non_hom_samples = [
    GraphInstance(
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        np.array([[1, 0, 0, 0, 0]], dtype=int),
        np.array([[0, 1]], dtype=int),
    ),
    GraphInstance(
        np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=int),
        np.array([[1, 2, 3, 4]], dtype=np.float32),
        np.array([[0, 1]], dtype=int),
    ),
    GraphInstance(
        np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=int),
        None,
        np.array([[0, 1]], dtype=int),
    ),
    (
        np.array([1, 0, 0, 0], dtype=int),
        np.array([0, 1, 0, 0], dtype=int),
        np.array([0, 0, 1, 0], dtype=int),
    ),
    (
        np.array([0, 1, 2, 3], dtype=np.float32),
        np.array([4, 5, 6, 7], dtype=np.float32),
    ),
    (
        np.array([0, 1, 2, 3], dtype=np.float32),
        np.array([4, 5, 6, 7], dtype=np.float32),
    ),
    OrderedDict(
        [
            (
                "a",
                (
                    np.array([1, 0, 0, 0], dtype=int),
                    np.array([0, 1, 0, 0], dtype=int),
                    np.array([0, 0, 1, 0], dtype=int),
                ),
            ),
            ("b", np.array([0, 1, 2, 3], dtype=np.float32)),
        ]
    ),
    OrderedDict(
        [
            (
                "a",
                GraphInstance(
                    np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=int),
                    np.array([[1, 0, 0, 0]], dtype=int),
                    np.array([[0, 1]], dtype=int),
                ),
            ),
            ("b", np.array([0, 1, 2, 3], dtype=np.float32)),
        ]
    ),
    (
        (
            np.array([1, 0, 0, 0], dtype=int),
            np.array([0, 1, 0, 0], dtype=int),
            np.array([0, 0, 1, 0], dtype=int),
        ),
        np.array([0, 1, 2, 3], dtype=np.float32),
    ),
    (
        GraphInstance(
            np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
            np.array([[1, 0, 0, 0]], dtype=int),
            np.array([[0, 1]], dtype=int),
        ),
        np.array([0, 1, 2, 3], dtype=np.float32),
    ),
    (
        GraphInstance(
            np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.float32),
            np.array([[1, 0, 0, 0]]),
            np.array([[0, 1]]),
        ),
        GraphInstance(
            np.array([[8, 9, 10, 11], [12, 13, 14, 15]], dtype=np.float32),
            np.array([[0, 1, 0, 0]]),
            np.array([[0, 1]]),
        ),
    ),
    OrderedDict(
        [
            (
                "a",
                OrderedDict(
                    [
                        (
                            "a",
                            (
                                np.array([0, 1, 2, 3], dtype=np.float32),
                                np.array([4, 5, 6, 7], dtype=np.float32),
                            ),
                        ),
                        ("b", np.array([8, 9, 10, 11], dtype=np.float32)),
                    ]
                ),
            ),
            ("b", (np.array([12, 13, 14, 15], dtype=np.float32))),
        ]
    ),
    OrderedDict(
        [
            (
                "a",
                OrderedDict(
                    [
                        (
                            "a",
                            GraphInstance(
                                np.array(
                                    [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32
                                ),
                                None,
                                np.array([[0, 1]], dtype=int),
                            ),
                        ),
                        ("b", np.array([8, 9, 10, 11], dtype=np.float32)),
                    ]
                ),
            ),
            ("b", (np.array([12, 13, 14, 15], dtype=np.float32))),
        ]
    ),
]


@pytest.mark.parametrize(
    ["space", "sample", "expected_flattened_sample"],
    zip(
        homogeneous_spaces + non_homogenous_spaces,
        homogeneous_samples + non_homogenous_samples,
        expected_flattened_hom_samples + expected_flattened_non_hom_samples,
    ),
)
def test_flatten(space, sample, expected_flattened_sample):
    flattened_sample = utils.flatten(space, sample)
    flat_space = utils.flatten_space(space)

    assert sample in space
    assert flattened_sample in flat_space

    if space.is_np_flattenable:
        assert isinstance(flattened_sample, np.ndarray)
        assert flattened_sample.shape == expected_flattened_sample.shape
        assert flattened_sample.dtype == expected_flattened_sample.dtype
        assert np.all(flattened_sample == expected_flattened_sample)
    else:
        assert not isinstance(flattened_sample, np.ndarray)
        assert compare_nested(flattened_sample, expected_flattened_sample)


@pytest.mark.parametrize(
    ["space", "flattened_sample", "expected_sample"],
    zip(homogeneous_spaces, expected_flattened_hom_samples, homogeneous_samples),
)
def test_unflatten(space, flattened_sample, expected_sample):
    sample = utils.unflatten(space, flattened_sample)
    assert compare_nested(sample, expected_sample)


expected_flattened_spaces = [
    Box(low=0, high=1, shape=(3,), dtype=np.int64),
    Box(low=0.0, high=np.inf, shape=(4,), dtype=np.float32),
    Box(low=0.0, high=np.inf, shape=(4,), dtype=np.float16),
    Box(low=0, high=1, shape=(15,), dtype=np.int64),
    Box(
        low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0], dtype=np.float64),
        dtype=np.float64,
    ),
    Box(low=0, high=1, shape=(9,), dtype=np.int64),
    Box(low=0, high=1, shape=(14,), dtype=np.int64),
    Box(low=0, high=1, shape=(10,), dtype=np.int8),
    Box(
        low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0], dtype=np.float64),
        dtype=np.float64,
    ),
    Box(low=0, high=1, shape=(3,), dtype=np.int64),
    Box(low=0, high=1, shape=(8,), dtype=np.int64),
]


@pytest.mark.parametrize(
    ["space", "expected_flattened_space"],
    zip(homogeneous_spaces, expected_flattened_spaces),
)
def test_flatten_space(space, expected_flattened_space):
    flattened_space = utils.flatten_space(space)
    assert flattened_space == expected_flattened_space
