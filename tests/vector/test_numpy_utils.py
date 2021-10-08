import itertools
from typing import Callable, Tuple as _Tuple, Optional
import pytest
import numpy as np

from collections import OrderedDict
from gym.spaces import Tuple, Dict, Space
from gym.vector.utils.spaces import _BaseGymSpaces
from tests.vector.utils import spaces

from gym.vector.utils.numpy_utils import concatenate, create_empty_array


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
@pytest.mark.parametrize("use_new_ordering", [True, False])
@pytest.mark.parametrize("n_pos_args", range(4))
def test_concatenate(space: Space, use_new_ordering: bool, n_pos_args: int):
    def assert_type(lhs, rhs, n):
        # Special case: if rhs is a list of scalars, lhs must be an np.ndarray
        if np.isscalar(rhs[0]):
            assert isinstance(lhs, np.ndarray)
            assert all([np.isscalar(rhs[i]) for i in range(n)])
        else:
            assert all([isinstance(rhs[i], type(lhs)) for i in range(n)])

    def assert_nested_equal(lhs, rhs, n):
        assert isinstance(rhs, list)
        assert (n > 0) and (len(rhs) == n)
        assert_type(lhs, rhs, n)
        if isinstance(lhs, np.ndarray):
            assert lhs.shape[0] == n
            for i in range(n):
                assert np.all(lhs[i] == rhs[i])

        elif isinstance(lhs, tuple):
            for i in range(len(lhs)):
                rhs_T_i = [rhs[j][i] for j in range(n)]
                assert_nested_equal(lhs[i], rhs_T_i, n)

        elif isinstance(lhs, OrderedDict):
            for key in lhs.keys():
                rhs_T_key = [rhs[j][key] for j in range(n)]
                assert_nested_equal(lhs[key], rhs_T_key, n)

        else:
            raise TypeError("Got unknown type `{0}`.".format(type(lhs)))

    items = [space.sample() for _ in range(8)]
    out = create_empty_array(space, n=8)
    n = 8
    # Use various ways of passing the arguments to the function. This is used to check for
    # backward-compatibility.
    if use_new_ordering:
        # Using new ordering: (space, items, out)
        # Test all combinations of positional / keyword arguments that make sense:
        keyword_args = OrderedDict([("space", space), ("items", items), ("out", out)])
    else:
        keyword_args = OrderedDict([("items", items), ("out", out), ("space", space)])
    
    # Take the first `n_pos_args` items out of `keyword_args` and into `positional_args`:
    positional_args = []
    for _ in range(n_pos_args):
        first_key = next(iter(keyword_args))
        positional_args.append(keyword_args.pop(first_key))

    # Call the function
    concatenated = concatenate(*positional_args, **keyword_args)
    
    assert np.all(concatenated == out)
    assert_nested_equal(out, items, n=8)


@pytest.mark.parametrize("n", [1, 8, None])
@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
@pytest.mark.parametrize("fn", [np.empty, np.zeros, np.ones])
@pytest.mark.parametrize("n_pos_args", range(4))
def test_create_empty_array(space: Space, n: Optional[int], n_pos_args: int, fn: Callable[..., np.ndarray]):
    
    def assert_nested_type(arr, space: Space, n: Optional[int]):
        if isinstance(space, _BaseGymSpaces):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == space.dtype
            if n is None:
                assert arr.shape == space.shape
            else:
                assert arr.shape == (n,) + space.shape
            if fn is np.zeros:
                assert np.all(arr == 0)
            elif fn is np.ones:
                assert np.all(arr == 1)

        elif isinstance(space, Tuple):
            assert isinstance(arr, tuple)
            assert len(arr) == len(space.spaces)
            for i in range(len(arr)):
                assert_nested_type(arr[i], space.spaces[i], n)

        elif isinstance(space, Dict):
            assert isinstance(arr, OrderedDict)
            assert set(arr.keys()) ^ set(space.spaces.keys()) == set()
            for key in arr.keys():
                assert_nested_type(arr[key], space.spaces[key], n)

        else:
            raise TypeError("Got unknown type `{0}`.".format(type(arr)))
    
    positional_args = []
    keyword_args = OrderedDict([("space", space), ("n", n), ("fn", fn)]) 
    # Take the first `n_pos_args` items out of `keyword_args` and into `positional_args`:
    for _ in range(n_pos_args):
        first_key = next(iter(keyword_args))
        positional_args.append(keyword_args.pop(first_key))

    array = create_empty_array(*positional_args, **keyword_args)
    assert_nested_type(array, space, n=n)
