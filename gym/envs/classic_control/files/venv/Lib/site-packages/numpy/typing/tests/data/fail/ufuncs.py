import numpy as np

np.sin.nin + "foo"  # E: Unsupported operand types
np.sin(1, foo="bar")  # E: Unexpected keyword argument
np.sin(1, extobj=["foo", "foo", "foo"])  # E: incompatible type

np.abs(None)  # E: incompatible type
