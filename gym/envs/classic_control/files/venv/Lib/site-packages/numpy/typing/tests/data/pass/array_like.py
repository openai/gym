from typing import Any, List, Optional

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, _SupportsArray

x1: ArrayLike = True
x2: ArrayLike = 5
x3: ArrayLike = 1.0
x4: ArrayLike = 1 + 1j
x5: ArrayLike = np.int8(1)
x6: ArrayLike = np.float64(1)
x7: ArrayLike = np.complex128(1)
x8: ArrayLike = np.array([1, 2, 3])
x9: ArrayLike = [1, 2, 3]
x10: ArrayLike = (1, 2, 3)
x11: ArrayLike = "foo"
x12: ArrayLike = memoryview(b'foo')


class A:
    def __array__(self, dtype: DTypeLike = None) -> np.ndarray:
        return np.array([1, 2, 3])


x13: ArrayLike = A()

scalar: _SupportsArray = np.int64(1)
scalar.__array__(np.float64)
array: _SupportsArray = np.array(1)
array.__array__(np.float64)

a: _SupportsArray = A()
a.__array__(np.int64)
a.__array__(dtype=np.int64)

# Escape hatch for when you mean to make something like an object
# array.
object_array_scalar: Any = (i for i in range(10))
np.array(object_array_scalar)
