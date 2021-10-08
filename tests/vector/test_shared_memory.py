import pytest
import numpy as np
from typing import Optional
import multiprocessing as mp
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing import Array, Process
from collections import OrderedDict

from gym.spaces import Tuple, Dict, Space
from gym.error import CustomSpaceError
from gym.vector.utils.spaces import _BaseGymSpaces
from tests.vector.utils import spaces, custom_spaces

from gym.vector.utils.shared_memory import (
    create_shared_memory,
    read_from_shared_memory,
    write_to_shared_memory,
)


expected_types = [
    Array("d", 1),
    Array("f", 1),
    Array("f", 3),
    Array("f", 4),
    Array("B", 1),
    Array("B", 32 * 32 * 3),
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
@pytest.mark.parametrize("n_pos_args", range(4))
def test_create_shared_memory(space, expected_type, n, ctx, n_pos_args: int):
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
            assert type(lhs[0]) == type(rhs[0])  # noqa: E721

        else:
            raise TypeError("Got unknown type `{0}`.".format(type(lhs)))

    ctx = mp if (ctx is None) else mp.get_context(ctx)

    positional_args = []
    keyword_args = OrderedDict(
        [
            ("space", space),
            ("n", n),
            ("ctx", ctx),
        ]
    )
    # Take the first `n_pos_args` items out of `keyword_args` and into `positional_args`:
    for _ in range(n_pos_args):
        first_key = next(iter(keyword_args))
        positional_args.append(keyword_args.pop(first_key))
    shared_memory = create_shared_memory(*positional_args, **keyword_args)
    assert_nested_type(shared_memory, expected_type, n=n)


@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize(
    "ctx", [None, "fork", "spawn"], ids=["default", "fork", "spawn"]
)
@pytest.mark.parametrize("space", custom_spaces)
@pytest.mark.parametrize("use_all_kwargs", [True, False])
def test_create_shared_memory_custom_space(
    n: int, ctx: Optional[str], space: Space, use_all_kwargs: bool
):
    ctx = mp if (ctx is None) else mp.get_context(ctx)
    with pytest.raises(CustomSpaceError):
        shared_memory = create_shared_memory(space, n=n, ctx=ctx)


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
@pytest.mark.parametrize("use_new_ordering", [True, False])
@pytest.mark.parametrize("n_pos_args", range(4))
def test_write_to_shared_memory(space: Space, use_new_ordering: bool, n_pos_args: int):
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
            raise TypeError("Got unknown type `{0}`.".format(type(lhs)))

    def write(i, shared_memory, sample):
        positional_args = []
        if use_new_ordering:
            keyword_args = OrderedDict(
                [
                    ("space", space),
                    ("index", i),
                    ("value", sample),
                    ("shared_memory", shared_memory),
                ]
            )
        else:
            # index, value, shared_memory, space
            keyword_args = OrderedDict(
                [
                    ("index", i),
                    ("value", sample),
                    ("shared_memory", shared_memory),
                    ("space", space),
                ]
            )

        # Take the first `n_pos_args` items out of `keyword_args` and into `positional_args`:
        for _ in range(n_pos_args):
            first_key = next(iter(keyword_args))
            positional_args.append(keyword_args.pop(first_key))

        write_to_shared_memory(*positional_args, **keyword_args)

    n = 8
    shared_memory = create_shared_memory(space, n=n)
    samples = [space.sample() for _ in range(n)]

    processes = [
        Process(target=write, args=(i, shared_memory, samples[i])) for i in range(n)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    assert_nested_equal(shared_memory, samples)


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
@pytest.mark.parametrize("use_new_ordering", [True, False])
@pytest.mark.parametrize("n_pos_args", range(4))
def test_read_from_shared_memory(space: Space, use_new_ordering: bool, n_pos_args: int):
    def assert_nested_equal(lhs, rhs, space: Space, n: int):
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

        elif isinstance(space, _BaseGymSpaces):
            assert isinstance(lhs, np.ndarray)
            assert lhs.shape == ((n,) + space.shape)
            assert lhs.dtype == space.dtype
            assert np.all(lhs == np.stack(rhs, axis=0))

        else:
            raise TypeError("Got unknown type `{0}`".format(type(space)))

    def write(i, shared_memory, sample):
        write_to_shared_memory(i, sample, shared_memory, space)

    n = 8
    shared_memory = create_shared_memory(space=space, n=n)

    positional_args = []
    if use_new_ordering:
        keyword_args = OrderedDict(
            [("space", space), ("shared_memory", shared_memory), ("n", n)]
        )
    else:
        keyword_args = OrderedDict(
            [("shared_memory", shared_memory), ("space", space), ("n", n)]
        )

    # Take the first `n_pos_args` items out of `keyword_args` and into `positional_args`:
    for _ in range(n_pos_args):
        first_key = next(iter(keyword_args))
        positional_args.append(keyword_args.pop(first_key))

    memory_view = read_from_shared_memory(*positional_args, **keyword_args)
    # array = create_empty_array(*positional_args, **keyword_args)

    samples = [space.sample() for _ in range(n)]

    processes = [
        Process(target=write, args=(i, shared_memory, samples[i])) for i in range(n)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    assert_nested_equal(memory_view, samples, space, n=n)
