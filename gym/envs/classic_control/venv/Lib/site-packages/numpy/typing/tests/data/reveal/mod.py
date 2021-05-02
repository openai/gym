import numpy as np

f8 = np.float64()
i8 = np.int64()
u8 = np.uint64()

f4 = np.float32()
i4 = np.int32()
u4 = np.uint32()

td = np.timedelta64(0, "D")
b_ = np.bool_()

b = bool()
f = float()
i = int()

AR = np.array([1], dtype=np.bool_)
AR.setflags(write=False)

AR2 = np.array([1], dtype=np.timedelta64)
AR2.setflags(write=False)

# Time structures

reveal_type(td % td)  # E: numpy.timedelta64
reveal_type(AR2 % td)  # E: Any
reveal_type(td % AR2)  # E: Any

reveal_type(divmod(td, td))  # E: Tuple[numpy.signedinteger[numpy.typing._64Bit], numpy.timedelta64]
reveal_type(divmod(AR2, td))  # E: Tuple[Any, Any]
reveal_type(divmod(td, AR2))  # E: Tuple[Any, Any]

# Bool

reveal_type(b_ % b)  # E: numpy.signedinteger[numpy.typing._8Bit]
reveal_type(b_ % i)  # E: numpy.signedinteger[Any]
reveal_type(b_ % f)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ % b_)  # E: numpy.signedinteger[numpy.typing._8Bit]
reveal_type(b_ % i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(b_ % u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(b_ % f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ % AR)  # E: Any

reveal_type(divmod(b_, b))  # E: Tuple[numpy.signedinteger[numpy.typing._8Bit], numpy.signedinteger[numpy.typing._8Bit]]
reveal_type(divmod(b_, i))  # E: Tuple[numpy.signedinteger[Any], numpy.signedinteger[Any]]
reveal_type(divmod(b_, f))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(b_, b_))  # E: Tuple[numpy.signedinteger[numpy.typing._8Bit], numpy.signedinteger[numpy.typing._8Bit]]
reveal_type(divmod(b_, i8))  # E: Tuple[numpy.signedinteger[numpy.typing._64Bit], numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(divmod(b_, u8))  # E: Tuple[numpy.unsignedinteger[numpy.typing._64Bit], numpy.unsignedinteger[numpy.typing._64Bit]]
reveal_type(divmod(b_, f8))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(b_, AR))  # E: Tuple[Any, Any]

reveal_type(b % b_)  # E: numpy.signedinteger[numpy.typing._8Bit]
reveal_type(i % b_)  # E: numpy.signedinteger[Any]
reveal_type(f % b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ % b_)  # E: numpy.signedinteger[numpy.typing._8Bit]
reveal_type(i8 % b_)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(u8 % b_)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(f8 % b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(AR % b_)  # E: Any

reveal_type(divmod(b, b_))  # E: Tuple[numpy.signedinteger[numpy.typing._8Bit], numpy.signedinteger[numpy.typing._8Bit]]
reveal_type(divmod(i, b_))  # E: Tuple[numpy.signedinteger[Any], numpy.signedinteger[Any]]
reveal_type(divmod(f, b_))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(b_, b_))  # E: Tuple[numpy.signedinteger[numpy.typing._8Bit], numpy.signedinteger[numpy.typing._8Bit]]
reveal_type(divmod(i8, b_))  # E: Tuple[numpy.signedinteger[numpy.typing._64Bit], numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(divmod(u8, b_))  # E: Tuple[numpy.unsignedinteger[numpy.typing._64Bit], numpy.unsignedinteger[numpy.typing._64Bit]]
reveal_type(divmod(f8, b_))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(AR, b_))  # E: Tuple[Any, Any]

# int

reveal_type(i8 % b)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 % i)  # E: numpy.signedinteger[Any]
reveal_type(i8 % f)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i8 % i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 % f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i4 % i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i4 % f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i4 % i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(i4 % f4)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(i8 % AR)  # E: Any

reveal_type(divmod(i8, b))  # E: Tuple[numpy.signedinteger[numpy.typing._64Bit], numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(divmod(i8, i))  # E: Tuple[numpy.signedinteger[Any], numpy.signedinteger[Any]]
reveal_type(divmod(i8, f))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(i8, i8))  # E: Tuple[numpy.signedinteger[numpy.typing._64Bit], numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(divmod(i8, f8))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(i8, i4))  # E: Tuple[numpy.signedinteger[numpy.typing._64Bit], numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(divmod(i8, f4))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(i4, i4))  # E: Tuple[numpy.signedinteger[numpy.typing._32Bit], numpy.signedinteger[numpy.typing._32Bit]]
reveal_type(divmod(i4, f4))  # E: Tuple[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._32Bit]]
reveal_type(divmod(i8, AR))  # E: Tuple[Any, Any]

reveal_type(b % i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i % i8)  # E: numpy.signedinteger[Any]
reveal_type(f % i8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i8 % i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(f8 % i8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i8 % i4)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(f8 % i4)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i4 % i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(f4 % i4)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(AR % i8)  # E: Any

reveal_type(divmod(b, i8))  # E: Tuple[numpy.signedinteger[numpy.typing._64Bit], numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(divmod(i, i8))  # E: Tuple[numpy.signedinteger[Any], numpy.signedinteger[Any]]
reveal_type(divmod(f, i8))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(i8, i8))  # E: Tuple[numpy.signedinteger[numpy.typing._64Bit], numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(divmod(f8, i8))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(i4, i8))  # E: Tuple[numpy.signedinteger[numpy.typing._64Bit], numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(divmod(f4, i8))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(i4, i4))  # E: Tuple[numpy.signedinteger[numpy.typing._32Bit], numpy.signedinteger[numpy.typing._32Bit]]
reveal_type(divmod(f4, i4))  # E: Tuple[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._32Bit]]
reveal_type(divmod(AR, i8))  # E: Tuple[Any, Any]

# float

reveal_type(f8 % b)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 % i)  # E: numpy.floating[Any]
reveal_type(f8 % f)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i8 % f4)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f4 % f4)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(f8 % AR)  # E: Any

reveal_type(divmod(f8, b))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(f8, i))  # E: Tuple[numpy.floating[Any], numpy.floating[Any]]
reveal_type(divmod(f8, f))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(f8, f8))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(f8, f4))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(f4, f4))  # E: Tuple[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._32Bit]]
reveal_type(divmod(f8, AR))  # E: Tuple[Any, Any]

reveal_type(b % f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i % f8)  # E: numpy.floating[Any]
reveal_type(f % f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 % f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 % f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f4 % f4)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(AR % f8)  # E: Any

reveal_type(divmod(b, f8))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(i, f8))  # E: Tuple[numpy.floating[Any], numpy.floating[Any]]
reveal_type(divmod(f, f8))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(f8, f8))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(f4, f8))  # E: Tuple[numpy.floating[numpy.typing._64Bit], numpy.floating[numpy.typing._64Bit]]
reveal_type(divmod(f4, f4))  # E: Tuple[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._32Bit]]
reveal_type(divmod(AR, f8))  # E: Tuple[Any, Any]
