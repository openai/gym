from typing import Callable
from gym.spaces.other.filter import FilteredSpace, Predicate

from gym.spaces import (
    Space,
    Box,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Dict as DictSpace,
    Tuple as TupleSpace,
)
from gym.spaces.space import T_cov

import pytest
import numpy as np


@pytest.mark.parametrize(
    "base_space, pred",
    [
        (Discrete(10), lambda v: v % 2 == 0,),
        (Box(0, 1, shape=(3,), dtype=np.float32), lambda v: (v > 0.5).all(),),
        (Box(0, 1, dtype=np.float32), lambda v: (v > 0.5).all(),),
        (
            TupleSpace([Discrete(5), Discrete(5, start=10)]),
            # note: Could be nice to auto-unpack the tuples if pred has >1 args.
            (lambda ab: ab[0] < 2 and ab[1] > 3),
        ),
        (
            TupleSpace(
                [
                    Discrete(5),
                    Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32),
                ]
            ),
            lambda a_array: a_array[0] > 2 and a_array[1].sum() > 2,
        ),
        (
            TupleSpace((Discrete(5), Discrete(2), Discrete(2))),
            (lambda a_b_c: sum(a_b_c) == 4),
        ),
        (
            DictSpace(
                {
                    "position": Discrete(5),
                    "velocity": Box(
                        low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32
                    ),
                }
            ),
            (lambda d: d["position"] > 2 and d["velocity"].sum() > 1.0),
        ),
    ],
)
@pytest.mark.parametrize("use_where", [False, True])
def test_samples_are_in_space(
    base_space: Space[T_cov], pred: Predicate[T_cov], use_where: bool
):
    space = (
        base_space.where(pred)
        if use_where
        else FilteredSpace(base_space, predicates=[pred])
    )
    n_samples = 20
    for _ in range(n_samples):
        sample = space.sample()
        assert sample in space
        assert pred(sample)


@pytest.mark.parametrize(
    "base_space, pred",
    [
        (Discrete(10), lambda v: v > 100),
        (Box(0, 1, shape=(10,), dtype=np.float32), lambda v: v.shape == (123,)),
        (Box(0, 1, dtype=np.float32), lambda v: (v < 0.0).all()),
    ],
)
@pytest.mark.parametrize("use_where", [False, True])
def test_sample_with_impossible_predicate_raises_error(
    base_space: Space[T_cov], pred: Predicate[T_cov], use_where: bool
):
    space = (
        base_space.where(pred)
        if use_where
        else FilteredSpace(base_space, predicates=[pred])
    )
    space.max_sampling_attempts = 100

    with pytest.raises(RuntimeError):
        space.sample()


@pytest.mark.parametrize(
    "base_space, pred",
    [
        (Discrete(10), lambda v: v % 2 == 0,),
        (Box(0, 1, shape=(3,), dtype=np.float32), lambda v: (0.1 <= v).all(),),
        (Box(0, 1, dtype=np.float32), lambda v: (v > 0.5).all(),),
        (
            TupleSpace([Discrete(5), Discrete(5, start=10)]),
            # note: Could be nice to auto-unpack the tuples if pred has >1 args.
            (lambda ab: ab[0] < 3 and ab[1] > 2),
        ),
        (
            TupleSpace(
                [
                    Discrete(5),
                    Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32),
                ]
            ),
            lambda a_array: a_array[0] > 2 and a_array[1].sum() > 1,
        ),
        (
            TupleSpace((Discrete(5), Discrete(2), Discrete(2))),
            (lambda a_b_c: sum(a_b_c) == 4),
        ),
        (
            DictSpace(
                {
                    "position": Discrete(5),
                    "velocity": Box(
                        low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32
                    ),
                }
            ),
            (lambda d: d["position"] > 2 and d["velocity"].sum() > 1.0),
        ),
    ],
)
@pytest.mark.parametrize("use_where", [False, True])
@pytest.mark.parametrize("n", [1, 3])
def test_batch_space_samples_are_in_space(
    base_space: Space[T_cov], pred: Predicate[T_cov], use_where: bool, n: int
):
    filtered_space = (
        base_space.where(pred)
        if use_where
        else FilteredSpace(base_space, predicates=[pred])
    )
    from gym.vector.utils.spaces import batch_space

    batched_base_space = batch_space(base_space, n=n)
    batched_filtered_space = batch_space(filtered_space, n=n)
    assert isinstance(batched_filtered_space, FilteredSpace)
    batched_filtered_space.max_sampling_attempts = 1000
    assert batched_filtered_space.space == batch_space(base_space, n=n)

    n_samples = 10
    for _ in range(n_samples):
        sample = batched_filtered_space.sample()
        assert sample in batched_base_space
        assert sample in batched_filtered_space

        assert all(predicate(sample) for predicate in batched_filtered_space.predicates)
