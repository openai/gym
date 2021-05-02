import sys
from typing import Any, List, Sequence, Tuple, Union, TYPE_CHECKING

from numpy import dtype
from ._shape import _ShapeLike

if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol, TypedDict
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

_DTypeLikeNested = Any  # TODO: wait for support for recursive types

if TYPE_CHECKING or HAVE_PROTOCOL:
    # Mandatory keys
    class _DTypeDictBase(TypedDict):
        names: Sequence[str]
        formats: Sequence[_DTypeLikeNested]

    # Mandatory + optional keys
    class _DTypeDict(_DTypeDictBase, total=False):
        offsets: Sequence[int]
        titles: Sequence[Any]  # Only `str` elements are usable as indexing aliases, but all objects are legal
        itemsize: int
        aligned: bool

    # A protocol for anything with the dtype attribute
    class _SupportsDType(Protocol):
        dtype: _DTypeLikeNested

else:
    _DTypeDict = Any
    _SupportsDType = Any


# Would create a dtype[np.void]
_VoidDTypeLike = Union[
    # (flexible_dtype, itemsize)
    Tuple[_DTypeLikeNested, int],
    # (fixed_dtype, shape)
    Tuple[_DTypeLikeNested, _ShapeLike],
    # [(field_name, field_dtype, field_shape), ...]
    #
    # The type here is quite broad because NumPy accepts quite a wide
    # range of inputs inside the list; see the tests for some
    # examples.
    List[Any],
    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ...,
    #  'itemsize': ...}
    _DTypeDict,
    # (base_dtype, new_dtype)
    Tuple[_DTypeLikeNested, _DTypeLikeNested],
]

# Anything that can be coerced into numpy.dtype.
# Reference: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
DTypeLike = Union[
    dtype,
    # default data type (float64)
    None,
    # array-scalar types and generic types
    type,  # TODO: enumerate these when we add type hints for numpy scalars
    # anything with a dtype attribute
    _SupportsDType,
    # character codes, type strings or comma-separated fields, e.g., 'float64'
    str,
    _VoidDTypeLike,
]

# NOTE: while it is possible to provide the dtype as a dict of
# dtype-like objects (e.g. `{'field1': ..., 'field2': ..., ...}`),
# this syntax is officially discourged and
# therefore not included in the Union defining `DTypeLike`.
#
# See https://github.com/numpy/numpy/issues/16891 for more details.
