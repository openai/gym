"""
A module with various ``typing.Protocol`` subclasses that implement
the ``__call__`` magic method.

See the `Mypy documentation`_ on protocols for more details.

.. _`Mypy documentation`: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols

"""

import sys
from typing import (
    Union,
    TypeVar,
    overload,
    Any,
    Tuple,
    NoReturn,
    TYPE_CHECKING,
)

from numpy import (
    generic,
    bool_,
    timedelta64,
    number,
    integer,
    unsignedinteger,
    signedinteger,
    int8,
    floating,
    float64,
    complexfloating,
    complex128,
)
from ._scalars import (
    _BoolLike,
    _IntLike,
    _FloatLike,
    _ComplexLike,
    _NumberLike,
)
from . import NBitBase

if sys.version_info >= (3, 8):
    from typing import Protocol
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

if TYPE_CHECKING or HAVE_PROTOCOL:
    _T = TypeVar("_T")
    _2Tuple = Tuple[_T, _T]

    _NBit_co = TypeVar("_NBit_co", covariant=True, bound=NBitBase)
    _NBit = TypeVar("_NBit", bound=NBitBase)

    _IntType = TypeVar("_IntType", bound=integer)
    _FloatType = TypeVar("_FloatType", bound=floating)
    _NumberType = TypeVar("_NumberType", bound=number)
    _NumberType_co = TypeVar("_NumberType_co", covariant=True, bound=number)
    _GenericType_co = TypeVar("_GenericType_co", covariant=True, bound=generic)

    class _BoolOp(Protocol[_GenericType_co]):
        @overload
        def __call__(self, __other: _BoolLike) -> _GenericType_co: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> signedinteger[Any]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(self, __other: _NumberType) -> _NumberType: ...

    class _BoolBitOp(Protocol[_GenericType_co]):
        @overload
        def __call__(self, __other: _BoolLike) -> _GenericType_co: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> signedinteger[Any]: ...
        @overload
        def __call__(self, __other: _IntType) -> _IntType: ...

    class _BoolSub(Protocol):
        # Note that `__other: bool_` is absent here
        @overload
        def __call__(self, __other: bool) -> NoReturn: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> signedinteger[Any]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(self, __other: _NumberType) -> _NumberType: ...

    class _BoolTrueDiv(Protocol):
        @overload
        def __call__(self, __other: Union[float, _IntLike, _BoolLike]) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(self, __other: _NumberType) -> _NumberType: ...

    class _BoolMod(Protocol):
        @overload
        def __call__(self, __other: _BoolLike) -> int8: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> signedinteger[Any]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: _IntType) -> _IntType: ...
        @overload
        def __call__(self, __other: _FloatType) -> _FloatType: ...

    class _BoolDivMod(Protocol):
        @overload
        def __call__(self, __other: _BoolLike) -> _2Tuple[int8]: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> _2Tuple[signedinteger[Any]]: ...
        @overload
        def __call__(self, __other: float) -> _2Tuple[float64]: ...
        @overload
        def __call__(self, __other: _IntType) -> _2Tuple[_IntType]: ...
        @overload
        def __call__(self, __other: _FloatType) -> _2Tuple[_FloatType]: ...

    class _TD64Div(Protocol[_NumberType_co]):
        @overload
        def __call__(self, __other: timedelta64) -> _NumberType_co: ...
        @overload
        def __call__(self, __other: _FloatLike) -> timedelta64: ...

    class _IntTrueDiv(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> floating[_NBit_co]: ...
        @overload
        def __call__(self, __other: int) -> floating[Any]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(self, __other: integer[_NBit]) -> floating[Union[_NBit_co, _NBit]]: ...

    class _UnsignedIntOp(Protocol[_NBit_co]):
        # NOTE: `uint64 + signedinteger -> float64`
        @overload
        def __call__(self, __other: bool) -> unsignedinteger[_NBit_co]: ...
        @overload
        def __call__(
            self, __other: Union[int, signedinteger[Any]]
        ) -> Union[signedinteger[Any], float64]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(
            self, __other: unsignedinteger[_NBit]
        ) -> unsignedinteger[Union[_NBit_co, _NBit]]: ...

    class _UnsignedIntBitOp(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> unsignedinteger[_NBit_co]: ...
        @overload
        def __call__(self, __other: int) -> signedinteger[Any]: ...
        @overload
        def __call__(self, __other: signedinteger[Any]) -> signedinteger[Any]: ...
        @overload
        def __call__(
            self, __other: unsignedinteger[_NBit]
        ) -> unsignedinteger[Union[_NBit_co, _NBit]]: ...

    class _UnsignedIntMod(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> unsignedinteger[_NBit_co]: ...
        @overload
        def __call__(
            self, __other: Union[int, signedinteger[Any]]
        ) -> Union[signedinteger[Any], float64]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(
            self, __other: unsignedinteger[_NBit]
        ) -> unsignedinteger[Union[_NBit_co, _NBit]]: ...

    class _UnsignedIntDivMod(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> _2Tuple[signedinteger[_NBit_co]]: ...
        @overload
        def __call__(
            self, __other: Union[int, signedinteger[Any]]
        ) -> Union[_2Tuple[signedinteger[Any]], _2Tuple[float64]]: ...
        @overload
        def __call__(self, __other: float) -> _2Tuple[float64]: ...
        @overload
        def __call__(
            self, __other: unsignedinteger[_NBit]
        ) -> _2Tuple[unsignedinteger[Union[_NBit_co, _NBit]]]: ...

    class _SignedIntOp(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> signedinteger[_NBit_co]: ...
        @overload
        def __call__(self, __other: int) -> signedinteger[Any]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(
            self, __other: signedinteger[_NBit]
        ) -> signedinteger[Union[_NBit_co, _NBit]]: ...

    class _SignedIntBitOp(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> signedinteger[_NBit_co]: ...
        @overload
        def __call__(self, __other: int) -> signedinteger[Any]: ...
        @overload
        def __call__(
            self, __other: signedinteger[_NBit]
        ) -> signedinteger[Union[_NBit_co, _NBit]]: ...

    class _SignedIntMod(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> signedinteger[_NBit_co]: ...
        @overload
        def __call__(self, __other: int) -> signedinteger[Any]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(
            self, __other: signedinteger[_NBit]
        ) -> signedinteger[Union[_NBit_co, _NBit]]: ...

    class _SignedIntDivMod(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> _2Tuple[signedinteger[_NBit_co]]: ...
        @overload
        def __call__(self, __other: int) -> _2Tuple[signedinteger[Any]]: ...
        @overload
        def __call__(self, __other: float) -> _2Tuple[float64]: ...
        @overload
        def __call__(
            self, __other: signedinteger[_NBit]
        ) -> _2Tuple[signedinteger[Union[_NBit_co, _NBit]]]: ...

    class _FloatOp(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> floating[_NBit_co]: ...
        @overload
        def __call__(self, __other: int) -> floating[Any]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(
            self, __other: Union[integer[_NBit], floating[_NBit]]
        ) -> floating[Union[_NBit_co, _NBit]]: ...

    class _FloatMod(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> floating[_NBit_co]: ...
        @overload
        def __call__(self, __other: int) -> floating[Any]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(
            self, __other: Union[integer[_NBit], floating[_NBit]]
        ) -> floating[Union[_NBit_co, _NBit]]: ...

    class _FloatDivMod(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> _2Tuple[floating[_NBit_co]]: ...
        @overload
        def __call__(self, __other: int) -> _2Tuple[floating[Any]]: ...
        @overload
        def __call__(self, __other: float) -> _2Tuple[float64]: ...
        @overload
        def __call__(
            self, __other: Union[integer[_NBit], floating[_NBit]]
        ) -> _2Tuple[floating[Union[_NBit_co, _NBit]]]: ...

    class _ComplexOp(Protocol[_NBit_co]):
        @overload
        def __call__(self, __other: bool) -> complexfloating[_NBit_co, _NBit_co]: ...
        @overload
        def __call__(self, __other: int) -> complexfloating[Any, Any]: ...
        @overload
        def __call__(self, __other: Union[float, complex]) -> complex128: ...
        @overload
        def __call__(
            self,
            __other: Union[
                integer[_NBit],
                floating[_NBit],
                complexfloating[_NBit, _NBit],
            ]
        ) -> complexfloating[Union[_NBit_co, _NBit], Union[_NBit_co, _NBit]]: ...

    class _NumberOp(Protocol):
        def __call__(self, __other: _NumberLike) -> number: ...

else:
    _BoolOp = Any
    _BoolBitOp = Any
    _BoolSub = Any
    _BoolTrueDiv = Any
    _BoolMod = Any
    _BoolDivMod = Any
    _TD64Div = Any
    _IntTrueDiv = Any
    _UnsignedIntOp = Any
    _UnsignedIntBitOp = Any
    _UnsignedIntMod = Any
    _UnsignedIntDivMod = Any
    _SignedIntOp = Any
    _SignedIntBitOp = Any
    _SignedIntMod = Any
    _SignedIntDivMod = Any
    _FloatOp = Any
    _FloatMod = Any
    _FloatDivMod = Any
    _ComplexOp = Any
    _NumberOp = Any
