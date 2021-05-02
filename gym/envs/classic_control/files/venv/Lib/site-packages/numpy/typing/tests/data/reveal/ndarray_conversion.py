import numpy as np

nd = np.array([[1, 2], [3, 4]])

# item
reveal_type(nd.item())  # E: Any
reveal_type(nd.item(1))  # E: Any
reveal_type(nd.item(0, 1))  # E: Any
reveal_type(nd.item((0, 1)))  # E: Any

# tolist
reveal_type(nd.tolist())  # E: Any

# itemset does not return a value
# tostring is pretty simple
# tobytes is pretty simple
# tofile does not return a value
# dump does not return a value
# dumps is pretty simple

# astype
reveal_type(nd.astype("float"))  # E: numpy.ndarray
reveal_type(nd.astype(float))  # E: numpy.ndarray
reveal_type(nd.astype(float, "K"))  # E: numpy.ndarray
reveal_type(nd.astype(float, "K", "unsafe"))  # E: numpy.ndarray
reveal_type(nd.astype(float, "K", "unsafe", True))  # E: numpy.ndarray
reveal_type(nd.astype(float, "K", "unsafe", True, True))  # E: numpy.ndarray

# byteswap
reveal_type(nd.byteswap())  # E: numpy.ndarray
reveal_type(nd.byteswap(True))  # E: numpy.ndarray

# copy
reveal_type(nd.copy())  # E: numpy.ndarray
reveal_type(nd.copy("C"))  # E: numpy.ndarray

# view
class SubArray(np.ndarray):
    pass


reveal_type(nd.view())  # E: numpy.ndarray
reveal_type(nd.view(np.int64))  # E: numpy.ndarray
# replace `Any` with `numpy.matrix` when `matrix` will be added to stubs
reveal_type(nd.view(np.int64, np.matrix))  # E: Any
reveal_type(nd.view(np.int64, SubArray))  # E: SubArray

# getfield
reveal_type(nd.getfield("float"))  # E: numpy.ndarray
reveal_type(nd.getfield(float))  # E: numpy.ndarray
reveal_type(nd.getfield(float, 8))  # E: numpy.ndarray

# setflags does not return a value
# fill does not return a value
