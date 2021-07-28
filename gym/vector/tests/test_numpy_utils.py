import pytest
import numpy as np

from collections import OrderedDict

from gym.spaces import Tuple, Dict
from gym.vector.utils.spaces import _BaseGymSpaces
from gym.vector.tests.utils import spaces

from gym.vector.utils.numpy_utils import concatenate, create_empty_array


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
def test_concatenate(space):
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

    samples = [space.sample() for _ in range(8)]
    array = create_empty_array(space, n=8)
    concatenated = concatenate(samples, array, space)

    assert np.all(concatenated == array)
    assert_nested_equal(array, samples, n=8)


@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
def test_create_empty_array(space, n):
    def assert_nested_type(arr, space, n):
        if isinstance(space, _BaseGymSpaces):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == space.dtype
            assert arr.shape == (n,) + space.shape

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

    array = create_empty_array(space, n=n, fn=np.empty)
    assert_nested_type(array, space, n=n)


@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
def test_create_empty_array_zeros(space, n):
    def assert_nested_type(arr, space, n):
        if isinstance(space, _BaseGymSpaces):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == space.dtype
            assert arr.shape == (n,) + space.shape
            assert np.all(arr == 0)

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

    array = create_empty_array(space, n=n, fn=np.zeros)
    assert_nested_type(array, space, n=n)


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
def test_create_empty_array_none_shape_ones(space):
    def assert_nested_type(arr, space):
        if isinstance(space, _BaseGymSpaces):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == space.dtype
            assert arr.shape == space.shape
            assert np.all(arr == 1)

        elif isinstance(space, Tuple):
            assert isinstance(arr, tuple)
            assert len(arr) == len(space.spaces)
            for i in range(len(arr)):
                assert_nested_type(arr[i], space.spaces[i])

        elif isinstance(space, Dict):
            assert isinstance(arr, OrderedDict)
            assert set(arr.keys()) ^ set(space.spaces.keys()) == set()
            for key in arr.keys():
                assert_nested_type(arr[key], space.spaces[key])

        else:
            raise TypeError("Got unknown type `{0}`.".format(type(arr)))

    array = create_empty_array(space, n=None, fn=np.ones)
    assert_nested_type(array, space)
