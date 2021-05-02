from typing import TypeVar, Union
import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=npt.NBitBase)

def add(a: np.floating[T], b: np.integer[T]) -> np.floating[T]:
    return a + b

i8: np.int64
i4: np.int32
f8: np.float64
f4: np.float32

reveal_type(add(f8, i8))  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(add(f4, i8))  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(add(f8, i4))  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(add(f4, i4))  # E: numpy.floating[numpy.typing._32Bit]
