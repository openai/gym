"""
Tests for miscellaneous (non-magic) ``np.ndarray``/``np.generic`` methods.

More extensive tests are performed for the methods'
function-based counterpart in `../from_numeric.py`.

"""

import numpy as np

class SubClass(np.ndarray): ...

f8: np.float64
A: np.ndarray
B: SubClass

reveal_type(f8.all())  # E: numpy.bool_
reveal_type(A.all())  # E: numpy.bool_
reveal_type(A.all(axis=0))  # E: Union[numpy.bool_, numpy.ndarray]
reveal_type(A.all(keepdims=True))  # E: Union[numpy.bool_, numpy.ndarray]
reveal_type(A.all(out=B))  # E: SubClass

reveal_type(f8.any())  # E: numpy.bool_
reveal_type(A.any())  # E: numpy.bool_
reveal_type(A.any(axis=0))  # E: Union[numpy.bool_, numpy.ndarray]
reveal_type(A.any(keepdims=True))  # E: Union[numpy.bool_, numpy.ndarray]
reveal_type(A.any(out=B))  # E: SubClass

reveal_type(f8.argmax())  # E: numpy.signedinteger[Any]
reveal_type(A.argmax())  # E: numpy.signedinteger[Any]
reveal_type(A.argmax(axis=0))  # E: Union[numpy.signedinteger[Any], numpy.ndarray]
reveal_type(A.argmax(out=B))  # E: SubClass

reveal_type(f8.argmin())  # E: numpy.signedinteger[Any]
reveal_type(A.argmin())  # E: numpy.signedinteger[Any]
reveal_type(A.argmin(axis=0))  # E: Union[numpy.signedinteger[Any], numpy.ndarray]
reveal_type(A.argmin(out=B))  # E: SubClass

reveal_type(f8.argsort())  # E: numpy.ndarray
reveal_type(A.argsort())  # E: numpy.ndarray

reveal_type(f8.astype(np.int64).choose([()]))  # E: numpy.ndarray
reveal_type(A.choose([0]))  # E: numpy.ndarray
reveal_type(A.choose([0], out=B))  # E: SubClass

reveal_type(f8.clip(1))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.clip(1))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.clip(None, 1))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.clip(1, out=B))  # E: SubClass
reveal_type(A.clip(None, 1, out=B))  # E: SubClass

reveal_type(f8.compress([0]))  # E: numpy.ndarray
reveal_type(A.compress([0]))  # E: numpy.ndarray
reveal_type(A.compress([0], out=B))  # E: SubClass

reveal_type(f8.conj())  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(A.conj())  # E: numpy.ndarray
reveal_type(B.conj())  # E: SubClass

reveal_type(f8.conjugate())  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(A.conjugate())  # E: numpy.ndarray
reveal_type(B.conjugate())  # E: SubClass

reveal_type(f8.cumprod())  # E: numpy.ndarray
reveal_type(A.cumprod())  # E: numpy.ndarray
reveal_type(A.cumprod(out=B))  # E: SubClass

reveal_type(f8.cumsum())  # E: numpy.ndarray
reveal_type(A.cumsum())  # E: numpy.ndarray
reveal_type(A.cumsum(out=B))  # E: SubClass

reveal_type(f8.max())  # E: numpy.number[Any]
reveal_type(A.max())  # E: numpy.number[Any]
reveal_type(A.max(axis=0))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.max(keepdims=True))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.max(out=B))  # E: SubClass

reveal_type(f8.mean())  # E: numpy.number[Any]
reveal_type(A.mean())  # E: numpy.number[Any]
reveal_type(A.mean(axis=0))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.mean(keepdims=True))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.mean(out=B))  # E: SubClass

reveal_type(f8.min())  # E: numpy.number[Any]
reveal_type(A.min())  # E: numpy.number[Any]
reveal_type(A.min(axis=0))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.min(keepdims=True))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.min(out=B))  # E: SubClass

reveal_type(f8.newbyteorder())  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(A.newbyteorder())  # E: numpy.ndarray
reveal_type(B.newbyteorder('|'))  # E: SubClass

reveal_type(f8.prod())  # E: numpy.number[Any]
reveal_type(A.prod())  # E: numpy.number[Any]
reveal_type(A.prod(axis=0))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.prod(keepdims=True))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.prod(out=B))  # E: SubClass

reveal_type(f8.ptp())  # E: numpy.number[Any]
reveal_type(A.ptp())  # E: numpy.number[Any]
reveal_type(A.ptp(axis=0))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.ptp(keepdims=True))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.ptp(out=B))  # E: SubClass

reveal_type(f8.round())  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(A.round())  # E: numpy.ndarray
reveal_type(A.round(out=B))  # E: SubClass

reveal_type(f8.repeat(1))  # E: numpy.ndarray
reveal_type(A.repeat(1))  # E: numpy.ndarray
reveal_type(B.repeat(1))  # E: numpy.ndarray

reveal_type(f8.std())  # E: numpy.number[Any]
reveal_type(A.std())  # E: numpy.number[Any]
reveal_type(A.std(axis=0))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.std(keepdims=True))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.std(out=B))  # E: SubClass

reveal_type(f8.sum())  # E: numpy.number[Any]
reveal_type(A.sum())  # E: numpy.number[Any]
reveal_type(A.sum(axis=0))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.sum(keepdims=True))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.sum(out=B))  # E: SubClass

reveal_type(f8.take(0))  # E: numpy.generic
reveal_type(A.take(0))  # E: numpy.generic
reveal_type(A.take([0]))  # E: numpy.ndarray
reveal_type(A.take(0, out=B))  # E: SubClass
reveal_type(A.take([0], out=B))  # E: SubClass

reveal_type(f8.var())  # E: numpy.number[Any]
reveal_type(A.var())  # E: numpy.number[Any]
reveal_type(A.var(axis=0))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.var(keepdims=True))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.var(out=B))  # E: SubClass

reveal_type(A.argpartition([0]))  # E: numpy.ndarray

reveal_type(A.diagonal())  # E: numpy.ndarray

reveal_type(A.dot(1))  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.dot(1, out=B))  # E: SubClass

reveal_type(A.nonzero())  # E: tuple[numpy.ndarray]

reveal_type(A.searchsorted([1]))  # E: numpy.ndarray

reveal_type(A.trace())  # E: Union[numpy.number[Any], numpy.ndarray]
reveal_type(A.trace(out=B))  # E: SubClass
