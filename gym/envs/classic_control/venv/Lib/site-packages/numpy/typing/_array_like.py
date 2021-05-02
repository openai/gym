import sys
from typing import Any, overload, Sequence, TYPE_CHECKING, Union

from numpy import ndarray
from ._scalars import _ScalarLike
from ._dtype_like import DTypeLike

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
    class _SupportsArray(Protocol):
        @overload
        def __array__(self, __dtype: DTypeLike = ...) -> ndarray: ...
        @overload
        def __array__(self, dtype: DTypeLike = ...) -> ndarray: ...
else:
    _SupportsArray = Any

# TODO: support buffer protocols once
#
# https://bugs.python.org/issue27501
#
# is resolved. See also the mypy issue:
#
# https://github.com/python/typing/issues/593
ArrayLike = Union[
    _ScalarLike,
    Sequence[_ScalarLike],
    Sequence[Sequence[Any]],  # TODO: Wait for support for recursive types
    _SupportsArray,
]
