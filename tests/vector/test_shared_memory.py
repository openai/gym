import multiprocessing as mp
from collections import OrderedDict
from multiprocessing import Array, Process
from multiprocessing.sharedctypes import SynchronizedArray

import numpy as np
import pytest

from gym.error import CustomSpaceError
from gym.spaces import Dict, Tuple
from gym.vector.utils.shared_memory import (
    create_shared_memory,
    read_from_shared_memory,
    write_to_shared_memory,
)
from gym.vector.utils.spaces import BaseGymSpaces
from tests.vector.utils import custom_spaces, spaces

expected_types = [
    Array("d", 1),
    Array("f", 1),
    Array("f", 3),
    Array("f", 4),
    Array("B", 1),
    Array("B", 32 * 32 * 3),
    Array("i", 1),
    Array("i", 1),
    (Array("i", 1), Array("i", 1)),
    (Array("i", 1), Array("f", 2)),
    Array("B", 3),
    Array("B", 19),
    OrderedDict([("position", Array("i", 1)), ("velocity", Array("f", 1))]),
    OrderedDict(
        [
            ("position", OrderedDict([("x", Array("i", 1)), ("y", Array("i", 1))])),
            ("velocity", (Array("i", 1), Array("B", 1))),
        ]
    ),
]


@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize(
    "space,expected_type",
    list(zip(spaces, expected_types)),
    ids=[space.__class__.__name__ for space in spaces],
)
@pytest.mark.parametrize(
    "ctx", [None, "fork", "spawn"], ids=["default", "fork", "spawn"]
)
def test_create_shared_memory(space, expected_type, n, ctx):
    def assert_nested_type(lhs, rhs, n):
        assert type(lhs) == type(rhs)
        if isinstance(lhs, (list, tuple)):
            assert len(lhs) == len(rhs)
            for lhs_, rhs_ in zip(lhs, rhs):
                assert_nested_type(lhs_, rhs_, n)

        elif isinstance(lhs, (dict, OrderedDict)):
            assert set(lhs.keys()) ^ set(rhs.keys()) == set()
            for key in lhs.keys():
                assert_nested_type(lhs[key], rhs[key], n)

        elif isinstance(lhs, SynchronizedArray):
            # Assert the length of the array
            assert len(lhs[:]) == n * len(rhs[:])
            # Assert the data type
            assert isinstance(lhs[0], type(rhs[0]))
        else:
            raise TypeError(f"Got unknown type `{type(lhs)}`.")

    ctx = mp if (ctx is None) else mp.get_context(ctx)
    shared_memory = create_shared_memory(space, n=n, ctx=ctx)
    assert_nested_type(shared_memory, expected_type, n=n)


@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize(
    "ctx", [None, "fork", "spawn"], ids=["default", "fork", "spawn"]
)
@pytest.mark.parametrize("space", custom_spaces)
def test_create_shared_memory_custom_space(n, ctx, space):
    ctx = mp if (ctx is None) else mp.get_context(ctx)
    with pytest.raises(CustomSpaceError):
        create_shared_memory(space, n=n, ctx=ctx)


def _write_shared_memory(space, i, shared_memory, sample):
    write_to_shared_memory(space, i, sample, shared_memory)


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
def test_write_to_shared_memory(space):
    def assert_nested_equal(lhs, rhs):
        assert isinstance(rhs, list)
        if isinstance(lhs, (list, tuple)):
            for i in range(len(lhs)):
                assert_nested_equal(lhs[i], [rhs_[i] for rhs_ in rhs])

        elif isinstance(lhs, (dict, OrderedDict)):
            for key in lhs.keys():
                assert_nested_equal(lhs[key], [rhs_[key] for rhs_ in rhs])

        elif isinstance(lhs, SynchronizedArray):
            assert np.all(np.array(lhs[:]) == np.stack(rhs, axis=0).flatten())

        else:
            raise TypeError(f"Got unknown type `{type(lhs)}`.")

    shared_memory_n8 = create_shared_memory(space, n=8)
    samples = [space.sample() for _ in range(8)]

    processes = [
        Process(
            target=_write_shared_memory, args=(space, i, shared_memory_n8, samples[i])
        )
        for i in range(8)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    assert_nested_equal(shared_memory_n8, samples)


def _process_write(space, i, shared_memory, sample):
    write_to_shared_memory(space, i, sample, shared_memory)


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
def test_read_from_shared_memory(space):
    def assert_nested_equal(lhs, rhs, space, n):
        assert isinstance(rhs, list)
        if isinstance(space, Tuple):
            assert isinstance(lhs, tuple)
            for i in range(len(lhs)):
                assert_nested_equal(
                    lhs[i], [rhs_[i] for rhs_ in rhs], space.spaces[i], n
                )

        elif isinstance(space, Dict):
            assert isinstance(lhs, OrderedDict)
            for key in lhs.keys():
                assert_nested_equal(
                    lhs[key], [rhs_[key] for rhs_ in rhs], space.spaces[key], n
                )

        elif isinstance(space, BaseGymSpaces):
            assert isinstance(lhs, np.ndarray)
            assert lhs.shape == ((n,) + space.shape)
            assert lhs.dtype == space.dtype
            assert np.all(lhs == np.stack(rhs, axis=0))

        else:
            raise TypeError(f"Got unknown type `{type(space)}`")

    shared_memory_n8 = create_shared_memory(space, n=8)
    memory_view_n8 = read_from_shared_memory(space, shared_memory_n8, n=8)
    samples = [space.sample() for _ in range(8)]

    processes = [
        Process(target=_process_write, args=(space, i, shared_memory_n8, samples[i]))
        for i in range(8)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    assert_nested_equal(memory_view_n8, samples, space, n=8)
