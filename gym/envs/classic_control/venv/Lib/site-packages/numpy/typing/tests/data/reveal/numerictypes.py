import numpy as np

reveal_type(np.issctype(np.generic))  # E: bool
reveal_type(np.issctype("foo"))  # E: bool

reveal_type(np.obj2sctype("S8"))  # E: Union[numpy.generic, None]
reveal_type(np.obj2sctype("S8", default=None))  # E: Union[numpy.generic, None]
reveal_type(
    np.obj2sctype("foo", default=int)  # E: Union[numpy.generic, Type[builtins.int*]]
)

reveal_type(np.issubclass_(np.float64, float))  # E: bool
reveal_type(np.issubclass_(np.float64, (int, float)))  # E: bool

reveal_type(np.sctype2char("S8"))  # E: str
reveal_type(np.sctype2char(list))  # E: str

reveal_type(np.find_common_type([np.int64], [np.int64]))  # E: numpy.dtype
