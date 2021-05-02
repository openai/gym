import numpy as np

np.maximum_sctype("S8")
np.maximum_sctype(object)

np.issctype(object)
np.issctype("S8")

np.obj2sctype(list)
np.obj2sctype(list, default=None)
np.obj2sctype(list, default=np.string_)

np.issubclass_(np.int32, int)
np.issubclass_(np.float64, float)
np.issubclass_(np.float64, (int, float))

np.issubsctype("int64", int)
np.issubsctype(np.array([1]), np.array([1]))

np.issubdtype("S1", np.string_)
np.issubdtype(np.float64, np.float32)

np.sctype2char("S1")
np.sctype2char(list)

np.find_common_type([], [np.int64, np.float32, complex])
np.find_common_type((), (np.int64, np.float32, complex))
np.find_common_type([np.int64, np.float32], [])
np.find_common_type([np.float32], [np.int64, np.float64])
