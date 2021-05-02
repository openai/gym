import sys
import datetime as dt
from typing import Optional, Union, Sequence, Tuple, Any, overload, TypeVar

from numpy import (
    ndarray,
    number,
    integer,
    bool_,
    generic,
    _OrderKACF,
    _OrderACF,
    _ArrayLikeBool,
    _ArrayLikeIntOrBool,
    _ModeKind,
    _PartitionKind,
    _SortKind,
    _SortSide,
)
from numpy.typing import (
    DTypeLike,
    ArrayLike,
    _ShapeLike,
    _Shape,
    _IntLike,
    _BoolLike,
    _NumberLike,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

# Various annotations for scalars

# While dt.datetime and dt.timedelta are not technically part of NumPy,
# they are one of the rare few builtin scalars which serve as valid return types.
# See https://github.com/numpy/numpy-stubs/pull/67#discussion_r412604113.
_ScalarNumpy = Union[generic, dt.datetime, dt.timedelta]
_ScalarBuiltin = Union[str, bytes, dt.date, dt.timedelta, bool, int, float, complex]
_Scalar = Union[_ScalarBuiltin, _ScalarNumpy]

# Integers and booleans can generally be used interchangeably
_ScalarIntOrBool = TypeVar("_ScalarIntOrBool", bound=Union[integer, bool_])
_ScalarGeneric = TypeVar("_ScalarGeneric", bound=generic)
_ScalarGenericDT = TypeVar(
    "_ScalarGenericDT", bound=Union[dt.datetime, dt.timedelta, generic]
)

_Number = TypeVar("_Number", bound=number)

# The signature of take() follows a common theme with its overloads:
# 1. A generic comes in; the same generic comes out
# 2. A scalar comes in; a generic comes out
# 3. An array-like object comes in; some keyword ensures that a generic comes out
# 4. An array-like object comes in; an ndarray or generic comes out
@overload
def take(
    a: _ScalarGenericDT,
    indices: int,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
    mode: _ModeKind = ...,
) -> _ScalarGenericDT: ...
@overload
def take(
    a: _Scalar,
    indices: int,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
    mode: _ModeKind = ...,
) -> _ScalarNumpy: ...
@overload
def take(
    a: ArrayLike,
    indices: int,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
    mode: _ModeKind = ...,
) -> _ScalarNumpy: ...
@overload
def take(
    a: ArrayLike,
    indices: _ArrayLikeIntOrBool,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
    mode: _ModeKind = ...,
) -> Union[_ScalarNumpy, ndarray]: ...
def reshape(a: ArrayLike, newshape: _ShapeLike, order: _OrderACF = ...) -> ndarray: ...
@overload
def choose(
    a: _ScalarIntOrBool,
    choices: ArrayLike,
    out: Optional[ndarray] = ...,
    mode: _ModeKind = ...,
) -> _ScalarIntOrBool: ...
@overload
def choose(
    a: Union[_IntLike, _BoolLike], choices: ArrayLike, out: Optional[ndarray] = ..., mode: _ModeKind = ...
) -> Union[integer, bool_]: ...
@overload
def choose(
    a: _ArrayLikeIntOrBool,
    choices: ArrayLike,
    out: Optional[ndarray] = ...,
    mode: _ModeKind = ...,
) -> ndarray: ...
def repeat(
    a: ArrayLike, repeats: _ArrayLikeIntOrBool, axis: Optional[int] = ...
) -> ndarray: ...
def put(
    a: ndarray, ind: _ArrayLikeIntOrBool, v: ArrayLike, mode: _ModeKind = ...
) -> None: ...
def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> ndarray: ...
def transpose(
    a: ArrayLike, axes: Union[None, Sequence[int], ndarray] = ...
) -> ndarray: ...
def partition(
    a: ArrayLike,
    kth: _ArrayLikeIntOrBool,
    axis: Optional[int] = ...,
    kind: _PartitionKind = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
@overload
def argpartition(
    a: generic,
    kth: _ArrayLikeIntOrBool,
    axis: Optional[int] = ...,
    kind: _PartitionKind = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> integer: ...
@overload
def argpartition(
    a: _ScalarBuiltin,
    kth: _ArrayLikeIntOrBool,
    axis: Optional[int] = ...,
    kind: _PartitionKind = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
@overload
def argpartition(
    a: ArrayLike,
    kth: _ArrayLikeIntOrBool,
    axis: Optional[int] = ...,
    kind: _PartitionKind = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
def sort(
    a: ArrayLike,
    axis: Optional[int] = ...,
    kind: Optional[_SortKind] = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
def argsort(
    a: ArrayLike,
    axis: Optional[int] = ...,
    kind: Optional[_SortKind] = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
@overload
def argmax(a: ArrayLike, axis: None = ..., out: Optional[ndarray] = ...) -> integer: ...
@overload
def argmax(
    a: ArrayLike, axis: int = ..., out: Optional[ndarray] = ...
) -> Union[integer, ndarray]: ...
@overload
def argmin(a: ArrayLike, axis: None = ..., out: Optional[ndarray] = ...) -> integer: ...
@overload
def argmin(
    a: ArrayLike, axis: int = ..., out: Optional[ndarray] = ...
) -> Union[integer, ndarray]: ...
@overload
def searchsorted(
    a: ArrayLike,
    v: _Scalar,
    side: _SortSide = ...,
    sorter: Optional[_ArrayLikeIntOrBool] = ...,  # 1D int array
) -> integer: ...
@overload
def searchsorted(
    a: ArrayLike,
    v: ArrayLike,
    side: _SortSide = ...,
    sorter: Optional[_ArrayLikeIntOrBool] = ...,  # 1D int array
) -> ndarray: ...
def resize(a: ArrayLike, new_shape: _ShapeLike) -> ndarray: ...
@overload
def squeeze(a: _ScalarGeneric, axis: Optional[_ShapeLike] = ...) -> _ScalarGeneric: ...
@overload
def squeeze(a: ArrayLike, axis: Optional[_ShapeLike] = ...) -> ndarray: ...
def diagonal(
    a: ArrayLike, offset: int = ..., axis1: int = ..., axis2: int = ...  # >= 2D array
) -> ndarray: ...
def trace(
    a: ArrayLike,  # >= 2D array
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
) -> Union[number, ndarray]: ...
def ravel(a: ArrayLike, order: _OrderKACF = ...) -> ndarray: ...
def nonzero(a: ArrayLike) -> Tuple[ndarray, ...]: ...
def shape(a: ArrayLike) -> _Shape: ...
def compress(
    condition: ArrayLike,  # 1D bool array
    a: ArrayLike,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
) -> ndarray: ...
@overload
def clip(
    a: _Number,
    a_min: ArrayLike,
    a_max: Optional[ArrayLike],
    out: Optional[ndarray] = ...,
    **kwargs: Any,
) -> _Number: ...
@overload
def clip(
    a: _Number,
    a_min: None,
    a_max: ArrayLike,
    out: Optional[ndarray] = ...,
    **kwargs: Any,
) -> _Number: ...
@overload
def clip(
    a: ArrayLike,
    a_min: ArrayLike,
    a_max: Optional[ArrayLike],
    out: Optional[ndarray] = ...,
    **kwargs: Any,
) -> Union[number, ndarray]: ...
@overload
def clip(
    a: ArrayLike,
    a_min: None,
    a_max: ArrayLike,
    out: Optional[ndarray] = ...,
    **kwargs: Any,
) -> Union[number, ndarray]: ...
@overload
def sum(
    a: _Number,
    axis: Optional[_ShapeLike] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> _Number: ...
@overload
def sum(
    a: ArrayLike,
    axis: _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> Union[number, ndarray]: ...
@overload
def all(
    a: ArrayLike,
    axis: None = ...,
    out: Optional[ndarray] = ...,
    keepdims: Literal[False] = ...,
) -> bool_: ...
@overload
def all(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Union[bool_, ndarray]: ...
@overload
def any(
    a: ArrayLike,
    axis: None = ...,
    out: Optional[ndarray] = ...,
    keepdims: Literal[False] = ...,
) -> bool_: ...
@overload
def any(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Union[bool_, ndarray]: ...
def cumsum(
    a: ArrayLike,
    axis: Optional[int] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
) -> ndarray: ...
@overload
def ptp(
    a: _Number,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> _Number: ...
@overload
def ptp(
    a: ArrayLike,
    axis: None = ...,
    out: Optional[ndarray] = ...,
    keepdims: Literal[False] = ...,
) -> number: ...
@overload
def ptp(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Union[number, ndarray]: ...
@overload
def amax(
    a: _Number,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> _Number: ...
@overload
def amax(
    a: ArrayLike,
    axis: None = ...,
    out: Optional[ndarray] = ...,
    keepdims: Literal[False] = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> number: ...
@overload
def amax(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> Union[number, ndarray]: ...
@overload
def amin(
    a: _Number,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> _Number: ...
@overload
def amin(
    a: ArrayLike,
    axis: None = ...,
    out: Optional[ndarray] = ...,
    keepdims: Literal[False] = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> number: ...
@overload
def amin(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> Union[number, ndarray]: ...

# TODO: `np.prod()``: For object arrays `initial` does not necessarily
# have to be a numerical scalar.
# The only requirement is that it is compatible
# with the `.__mul__()` method(s) of the passed array's elements.

# Note that the same situation holds for all wrappers around
# `np.ufunc.reduce`, e.g. `np.sum()` (`.__add__()`).
@overload
def prod(
    a: _Number,
    axis: Optional[_ShapeLike] = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> _Number: ...
@overload
def prod(
    a: ArrayLike,
    axis: None = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> number: ...
@overload
def prod(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> Union[number, ndarray]: ...
def cumprod(
    a: ArrayLike,
    axis: Optional[int] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
) -> ndarray: ...
def ndim(a: ArrayLike) -> int: ...
def size(a: ArrayLike, axis: Optional[int] = ...) -> int: ...
@overload
def around(
    a: _Number, decimals: int = ..., out: Optional[ndarray] = ...
) -> _Number: ...
@overload
def around(
    a: _NumberLike, decimals: int = ..., out: Optional[ndarray] = ...
) -> number: ...
@overload
def around(
    a: ArrayLike, decimals: int = ..., out: Optional[ndarray] = ...
) -> ndarray: ...
@overload
def mean(
    a: ArrayLike,
    axis: None = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
) -> number: ...
@overload
def mean(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Union[number, ndarray]: ...
@overload
def std(
    a: ArrayLike,
    axis: None = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    ddof: int = ...,
    keepdims: Literal[False] = ...,
) -> number: ...
@overload
def std(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    ddof: int = ...,
    keepdims: bool = ...,
) -> Union[number, ndarray]: ...
@overload
def var(
    a: ArrayLike,
    axis: None = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    ddof: int = ...,
    keepdims: Literal[False] = ...,
) -> number: ...
@overload
def var(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DTypeLike = ...,
    out: Optional[ndarray] = ...,
    ddof: int = ...,
    keepdims: bool = ...,
) -> Union[number, ndarray]: ...
