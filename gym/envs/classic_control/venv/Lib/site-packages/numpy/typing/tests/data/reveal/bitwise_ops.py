import numpy as np

i8 = np.int64(1)
u8 = np.uint64(1)

i4 = np.int32(1)
u4 = np.uint32(1)

b_ = np.bool_(1)

b = bool(1)
i = int(1)

AR = np.array([0, 1, 2], dtype=np.int32)
AR.setflags(write=False)


reveal_type(i8 << i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 >> i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 | i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 ^ i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 & i8)  # E: numpy.signedinteger[numpy.typing._64Bit]

reveal_type(i8 << AR)  # E: Any
reveal_type(i8 >> AR)  # E: Any
reveal_type(i8 | AR)  # E: Any
reveal_type(i8 ^ AR)  # E: Any
reveal_type(i8 & AR)  # E: Any

reveal_type(i4 << i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(i4 >> i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(i4 | i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(i4 ^ i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(i4 & i4)  # E: numpy.signedinteger[numpy.typing._32Bit]

reveal_type(i8 << i4)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 >> i4)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 | i4)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 ^ i4)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 & i4)  # E: numpy.signedinteger[numpy.typing._64Bit]

reveal_type(i8 << i)  # E: numpy.signedinteger[Any]
reveal_type(i8 >> i)  # E: numpy.signedinteger[Any]
reveal_type(i8 | i)  # E: numpy.signedinteger[Any]
reveal_type(i8 ^ i)  # E: numpy.signedinteger[Any]
reveal_type(i8 & i)  # E: numpy.signedinteger[Any]

reveal_type(i8 << b_)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 >> b_)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 | b_)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 ^ b_)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 & b_)  # E: numpy.signedinteger[numpy.typing._64Bit]

reveal_type(i8 << b)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 >> b)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 | b)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 ^ b)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 & b)  # E: numpy.signedinteger[numpy.typing._64Bit]

reveal_type(u8 << u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 >> u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 | u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 ^ u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 & u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]

reveal_type(u8 << AR)  # E: Any
reveal_type(u8 >> AR)  # E: Any
reveal_type(u8 | AR)  # E: Any
reveal_type(u8 ^ AR)  # E: Any
reveal_type(u8 & AR)  # E: Any

reveal_type(u4 << u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(u4 >> u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(u4 | u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(u4 ^ u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(u4 & u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]

reveal_type(u4 << i4)  # E: numpy.signedinteger[Any]
reveal_type(u4 >> i4)  # E: numpy.signedinteger[Any]
reveal_type(u4 | i4)  # E: numpy.signedinteger[Any]
reveal_type(u4 ^ i4)  # E: numpy.signedinteger[Any]
reveal_type(u4 & i4)  # E: numpy.signedinteger[Any]

reveal_type(u4 << i)  # E: numpy.signedinteger[Any]
reveal_type(u4 >> i)  # E: numpy.signedinteger[Any]
reveal_type(u4 | i)  # E: numpy.signedinteger[Any]
reveal_type(u4 ^ i)  # E: numpy.signedinteger[Any]
reveal_type(u4 & i)  # E: numpy.signedinteger[Any]

reveal_type(u8 << b_)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 >> b_)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 | b_)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 ^ b_)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 & b_)  # E: numpy.unsignedinteger[numpy.typing._64Bit]

reveal_type(u8 << b)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 >> b)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 | b)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 ^ b)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 & b)  # E: numpy.unsignedinteger[numpy.typing._64Bit]

reveal_type(b_ << b_)  # E: numpy.signedinteger[numpy.typing._8Bit]
reveal_type(b_ >> b_)  # E: numpy.signedinteger[numpy.typing._8Bit]
reveal_type(b_ | b_)  # E: numpy.bool_
reveal_type(b_ ^ b_)  # E: numpy.bool_
reveal_type(b_ & b_)  # E: numpy.bool_

reveal_type(b_ << AR)  # E: Any
reveal_type(b_ >> AR)  # E: Any
reveal_type(b_ | AR)  # E: Any
reveal_type(b_ ^ AR)  # E: Any
reveal_type(b_ & AR)  # E: Any

reveal_type(b_ << b)  # E: numpy.signedinteger[numpy.typing._8Bit]
reveal_type(b_ >> b)  # E: numpy.signedinteger[numpy.typing._8Bit]
reveal_type(b_ | b)  # E: numpy.bool_
reveal_type(b_ ^ b)  # E: numpy.bool_
reveal_type(b_ & b)  # E: numpy.bool_

reveal_type(b_ << i)  # E: numpy.signedinteger[Any]
reveal_type(b_ >> i)  # E: numpy.signedinteger[Any]
reveal_type(b_ | i)  # E: numpy.signedinteger[Any]
reveal_type(b_ ^ i)  # E: numpy.signedinteger[Any]
reveal_type(b_ & i)  # E: numpy.signedinteger[Any]

reveal_type(~i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(~i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(~u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(~u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(~b_)  # E: numpy.bool_
reveal_type(~AR)  # E: Any
