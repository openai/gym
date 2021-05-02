"""
The tests exercise the casting machinery in a more low-level manner.
The reason is mostly to test a new implementation of the casting machinery.

Unlike most tests in NumPy, these are closer to unit-tests rather
than integration tests.
"""

import pytest
import textwrap
import enum

import numpy as np

from numpy.core._multiarray_umath import (
    _get_castingimpl as get_castingimpl)
from numpy.core._multiarray_tests import uses_new_casts


# Simple skips object, parametric and long double (unsupported by struct)
simple_dtypes = "?bhilqBHILQefdFD"
if np.dtype("l").itemsize != np.dtype("q").itemsize:
    # Remove l and L, the table was generated with 64bit linux in mind.
    # TODO: Should have two tables or no a different solution.
    simple_dtypes = simple_dtypes.replace("l", "").replace("L", "")
simple_dtypes = [type(np.dtype(c)) for c in simple_dtypes]


def simple_dtype_instances():
    for dtype_class in simple_dtypes:
        dt = dtype_class()
        yield pytest.param(dt, id=str(dt))
        if dt.byteorder != "|":
            dt = dt.newbyteorder()
            yield pytest.param(dt, id=str(dt))


def get_expected_stringlength(dtype):
    """Returns the string length when casting the basic dtypes to strings.
    """
    if dtype == np.bool_:
        return 5
    if dtype.kind in "iu":
        if dtype.itemsize == 1:
            length = 3
        elif dtype.itemsize == 2:
            length = 5
        elif dtype.itemsize == 4:
            length = 10
        elif dtype.itemsize == 8:
            length = 20
        else:
            raise AssertionError(f"did not find expected length for {dtype}")

        if dtype.kind == "i":
            length += 1  # adds one character for the sign

        return length

    # Note: Can't do dtype comparison for longdouble on windows
    if dtype.char == "g":
        return 48
    elif dtype.char == "G":
        return 48 * 2
    elif dtype.kind == "f":
        return 32  # also for half apparently.
    elif dtype.kind == "c":
        return 32 * 2

    raise AssertionError(f"did not find expected length for {dtype}")


class Casting(enum.IntEnum):
    no = 0
    equiv = 1
    safe = 2
    same_kind = 3
    unsafe = 4
    cast_is_view = 1 << 16


def _get_cancast_table():
    table = textwrap.dedent("""
        X ? b h i l q B H I L Q e f d g F D G S U V O M m
        ? # = = = = = = = = = = = = = = = = = = = = = . =
        b . # = = = = . . . . . = = = = = = = = = = = . =
        h . ~ # = = = . . . . . ~ = = = = = = = = = = . =
        i . ~ ~ # = = . . . . . ~ ~ = = ~ = = = = = = . =
        l . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
        q . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
        B . ~ = = = = # = = = = = = = = = = = = = = = . =
        H . ~ ~ = = = ~ # = = = ~ = = = = = = = = = = . =
        I . ~ ~ ~ = = ~ ~ # = = ~ ~ = = ~ = = = = = = . =
        L . ~ ~ ~ ~ ~ ~ ~ ~ # # ~ ~ = = ~ = = = = = = . ~
        Q . ~ ~ ~ ~ ~ ~ ~ ~ # # ~ ~ = = ~ = = = = = = . ~
        e . . . . . . . . . . . # = = = = = = = = = = . .
        f . . . . . . . . . . . ~ # = = = = = = = = = . .
        d . . . . . . . . . . . ~ ~ # = ~ = = = = = = . .
        g . . . . . . . . . . . ~ ~ ~ # ~ ~ = = = = = . .
        F . . . . . . . . . . . . . . . # = = = = = = . .
        D . . . . . . . . . . . . . . . ~ # = = = = = . .
        G . . . . . . . . . . . . . . . ~ ~ # = = = = . .
        S . . . . . . . . . . . . . . . . . . # = = = . .
        U . . . . . . . . . . . . . . . . . . . # = = . .
        V . . . . . . . . . . . . . . . . . . . . # = . .
        O . . . . . . . . . . . . . . . . . . . . = # . .
        M . . . . . . . . . . . . . . . . . . . . = = # .
        m . . . . . . . . . . . . . . . . . . . . = = . #
        """).strip().split("\n")
    dtypes = [type(np.dtype(c)) for c in table[0][2::2]]

    convert_cast = {".": Casting.unsafe, "~": Casting.same_kind,
                    "=": Casting.safe, "#": Casting.equiv,
                    " ": -1}

    cancast = {}
    for from_dt, row in zip(dtypes, table[1:]):
        cancast[from_dt] = {}
        for to_dt, c in zip(dtypes, row[2::2]):
            cancast[from_dt][to_dt] = convert_cast[c]

    return cancast

CAST_TABLE = _get_cancast_table()


class TestChanges:
    """
    These test cases excercise some behaviour changes
    """
    @pytest.mark.parametrize("string", ["S", "U"])
    @pytest.mark.parametrize("floating", ["e", "f", "d", "g"])
    def test_float_to_string(self, floating, string):
        assert np.can_cast(floating, string)
        # 100 is long enough to hold any formatted floating
        if uses_new_casts():
            assert np.can_cast(floating, f"{string}100")
        else:
            assert not np.can_cast(floating, f"{string}100")
            assert np.can_cast(floating, f"{string}100", casting="same_kind")

    def test_to_void(self):
        # But in general, we do consider these safe:
        assert np.can_cast("d", "V")
        assert np.can_cast("S20", "V")

        # Do not consider it a safe cast if the void is too smaller:
        if uses_new_casts():
            assert not np.can_cast("d", "V1")
            assert not np.can_cast("S20", "V1")
            assert not np.can_cast("U1", "V1")
            # Structured to unstructured is just like any other:
            assert np.can_cast("d,i", "V", casting="same_kind")
        else:
            assert np.can_cast("d", "V1")
            assert np.can_cast("S20", "V1")
            assert np.can_cast("U1", "V1")
            assert not np.can_cast("d,i", "V", casting="same_kind")


class TestCasting:
    @pytest.mark.parametrize("from_Dt", simple_dtypes)
    def test_simple_cancast(self, from_Dt):
        for to_Dt in simple_dtypes:
            cast = get_castingimpl(from_Dt, to_Dt)

            for from_dt in [from_Dt(), from_Dt().newbyteorder()]:
                default = cast._resolve_descriptors((from_dt, None))[1][1]
                assert default == to_Dt()
                del default

                for to_dt in [to_Dt(), to_Dt().newbyteorder()]:
                    casting, (from_res, to_res) = cast._resolve_descriptors(
                        (from_dt, to_dt))
                    assert(type(from_res) == from_Dt)
                    assert(type(to_res) == to_Dt)
                    if casting & Casting.cast_is_view:
                        # If a view is acceptable, this is "no" casting
                        # and byte order must be matching.
                        assert casting == Casting.no | Casting.cast_is_view
                        # The above table lists this as "equivalent"
                        assert Casting.equiv == CAST_TABLE[from_Dt][to_Dt]
                        # Note that to_res may not be the same as from_dt
                        assert from_res.isnative == to_res.isnative
                    else:
                        if from_Dt == to_Dt:
                            # Note that to_res may not be the same as from_dt
                            assert from_res.isnative != to_res.isnative
                        assert casting == CAST_TABLE[from_Dt][to_Dt]

                    if from_Dt is to_Dt:
                        assert(from_dt is from_res)
                        assert(to_dt is to_res)


    def string_with_modified_length(self, dtype, change_length):
        fact = 1 if dtype.char == "S" else 4
        length = dtype.itemsize // fact + change_length
        return np.dtype(f"{dtype.byteorder}{dtype.char}{length}")

    @pytest.mark.parametrize("other_DT", simple_dtypes)
    @pytest.mark.parametrize("string_char", ["S", "U"])
    def test_string_cancast(self, other_DT, string_char):
        fact = 1 if string_char == "S" else 4

        string_DT = type(np.dtype(string_char))
        cast = get_castingimpl(other_DT, string_DT)

        other_dt = other_DT()
        expected_length = get_expected_stringlength(other_dt)
        string_dt = np.dtype(f"{string_char}{expected_length}")

        safety, (res_other_dt, res_dt) = cast._resolve_descriptors((other_dt, None))
        assert res_dt.itemsize == expected_length * fact
        assert safety == Casting.safe  # we consider to string casts "safe"
        assert isinstance(res_dt, string_DT)

        # These casts currently implement changing the string length, so
        # check the cast-safety for too long/fixed string lengths:
        for change_length in [-1, 0, 1]:
            if change_length >= 0:
                expected_safety = Casting.safe
            else:
                expected_safety = Casting.same_kind

            to_dt = self.string_with_modified_length(string_dt, change_length)
            safety, (_, res_dt) = cast._resolve_descriptors((other_dt, to_dt))
            assert res_dt is to_dt
            assert safety == expected_safety

        # The opposite direction is always considered unsafe:
        cast = get_castingimpl(string_DT, other_DT)

        safety, _ = cast._resolve_descriptors((string_dt, other_dt))
        assert safety == Casting.unsafe

        cast = get_castingimpl(string_DT, other_DT)
        safety, (_, res_dt) = cast._resolve_descriptors((string_dt, None))
        assert safety == Casting.unsafe
        assert other_dt is res_dt  # returns the singleton for simple dtypes

    @pytest.mark.parametrize("other_dt", ["S8", "<U8", ">U8"])
    @pytest.mark.parametrize("string_char", ["S", "U"])
    def test_string_to_string_cancast(self, other_dt, string_char):
        other_dt = np.dtype(other_dt)

        fact = 1 if string_char == "S" else 4
        div = 1 if other_dt.char == "S" else 4

        string_DT = type(np.dtype(string_char))
        cast = get_castingimpl(type(other_dt), string_DT)

        expected_length = other_dt.itemsize // div
        string_dt = np.dtype(f"{string_char}{expected_length}")

        safety, (res_other_dt, res_dt) = cast._resolve_descriptors((other_dt, None))
        assert res_dt.itemsize == expected_length * fact
        assert isinstance(res_dt, string_DT)

        if other_dt.char == string_char:
            if other_dt.isnative:
                expected_safety = Casting.no | Casting.cast_is_view
            else:
                expected_safety = Casting.equiv
        elif string_char == "U":
            expected_safety = Casting.safe
        else:
            expected_safety = Casting.unsafe

        assert expected_safety == safety

        for change_length in [-1, 0, 1]:
            to_dt = self.string_with_modified_length(string_dt, change_length)
            safety, (_, res_dt) = cast._resolve_descriptors((other_dt, to_dt))

            assert res_dt is to_dt
            if expected_safety == Casting.unsafe:
                assert safety == expected_safety
            elif change_length < 0:
                assert safety == Casting.same_kind
            elif change_length == 0:
                assert safety == expected_safety
            elif change_length > 0:
                assert safety == Casting.safe

    def test_void_to_string_special_case(self):
        # Cover a small special case in void to string casting that could
        # probably just as well be turned into an error (compare
        # `test_object_to_parametric_internal_error` below).
        assert np.array([], dtype="V5").astype("S").dtype.itemsize == 5
        assert np.array([], dtype="V5").astype("U").dtype.itemsize == 4 * 5

    def test_object_to_parametric_internal_error(self):
        # We reject casting from object to a parametric type, without
        # figuring out the correct instance first.
        object_dtype = type(np.dtype(object))
        other_dtype = type(np.dtype(str))
        cast = get_castingimpl(object_dtype, other_dtype)
        with pytest.raises(TypeError,
                    match="casting from object to the parametric DType"):
            cast._resolve_descriptors((np.dtype("O"), None))
