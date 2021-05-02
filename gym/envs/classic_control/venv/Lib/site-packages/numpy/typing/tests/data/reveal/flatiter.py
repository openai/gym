import numpy as np

a: "np.flatiter[np.ndarray]"

reveal_type(a.base)  # E: numpy.ndarray*
reveal_type(a.copy())  # E: numpy.ndarray*
reveal_type(a.coords)  # E: tuple[builtins.int]
reveal_type(a.index)  # E: int
reveal_type(iter(a))  # E: Iterator[numpy.generic*]
reveal_type(next(a))  # E: numpy.generic
reveal_type(a[0])  # E: numpy.generic
reveal_type(a[[0, 1, 2]])  # E: numpy.ndarray*
reveal_type(a[...])  # E: numpy.ndarray*
reveal_type(a[:])  # E: numpy.ndarray*
