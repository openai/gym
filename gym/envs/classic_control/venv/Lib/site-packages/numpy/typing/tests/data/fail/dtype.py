import numpy as np

class Test:
    not_dtype = float


np.dtype(Test())  # E: No overload variant of "dtype" matches

np.dtype(  # E: No overload variant of "dtype" matches
    {
        "field1": (float, 1),
        "field2": (int, 3),
    }
)

np.dtype[np.float64](np.int64)  # E: Argument 1 to "dtype" has incompatible type
