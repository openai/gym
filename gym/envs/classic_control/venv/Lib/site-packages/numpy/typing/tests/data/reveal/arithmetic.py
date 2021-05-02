import numpy as np

c16 = np.complex128()
f8 = np.float64()
i8 = np.int64()
u8 = np.uint64()

c8 = np.complex64()
f4 = np.float32()
i4 = np.int32()
u4 = np.uint32()

dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

b_ = np.bool_()

b = bool()
c = complex()
f = float()
i = int()

AR = np.array([0], dtype=np.float64)
AR.setflags(write=False)

# unary ops

reveal_type(-c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(-c8)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(-f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(-f4)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(-i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(-i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(-u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(-u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(-td)  # E: numpy.timedelta64
reveal_type(-AR)  # E: Any

reveal_type(+c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(+c8)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(+f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(+f4)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(+i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(+i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(+u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(+u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(+td)  # E: numpy.timedelta64
reveal_type(+AR)  # E: Any

reveal_type(abs(c16))  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(abs(c8))  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(abs(f8))  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(abs(f4))  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(abs(i8))  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(abs(i4))  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(abs(u8))  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(abs(u4))  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(abs(td))  # E: numpy.timedelta64
reveal_type(abs(b_))  # E: numpy.bool_
reveal_type(abs(AR))  # E: Any

# Time structures

reveal_type(dt + td)  # E: numpy.datetime64
reveal_type(dt + i)  # E: numpy.datetime64
reveal_type(dt + i4)  # E: numpy.datetime64
reveal_type(dt + i8)  # E: numpy.datetime64
reveal_type(dt - dt)  # E: numpy.timedelta64
reveal_type(dt - i)  # E: numpy.datetime64
reveal_type(dt - i4)  # E: numpy.datetime64
reveal_type(dt - i8)  # E: numpy.datetime64

reveal_type(td + td)  # E: numpy.timedelta64
reveal_type(td + i)  # E: numpy.timedelta64
reveal_type(td + i4)  # E: numpy.timedelta64
reveal_type(td + i8)  # E: numpy.timedelta64
reveal_type(td - td)  # E: numpy.timedelta64
reveal_type(td - i)  # E: numpy.timedelta64
reveal_type(td - i4)  # E: numpy.timedelta64
reveal_type(td - i8)  # E: numpy.timedelta64
reveal_type(td / f)  # E: numpy.timedelta64
reveal_type(td / f4)  # E: numpy.timedelta64
reveal_type(td / f8)  # E: numpy.timedelta64
reveal_type(td / td)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(td // td)  # E: numpy.signedinteger[numpy.typing._64Bit]

# boolean

reveal_type(b_ / b)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ / b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ / i)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ / i8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ / i4)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ / u8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ / u4)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ / f)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ / f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ / f4)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(b_ / c)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(b_ / c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(b_ / c8)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]

reveal_type(b / b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ / b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i / b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i8 / b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i4 / b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(u8 / b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(u4 / b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f / b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 / b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f4 / b_)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(c / b_)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 / b_)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c8 / b_)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]

# Complex

reveal_type(c16 + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 + f8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 + i8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 + c8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 + f4)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 + i4)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 + b_)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 + b)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 + c)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 + f)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c16 + i)  # E: numpy.complexfloating[Any, Any]
reveal_type(c16 + AR)  # E: Any

reveal_type(c16 + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f8 + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(i8 + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c8 + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f4 + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(i4 + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(b_ + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(b + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(i + c16)  # E: numpy.complexfloating[Any, Any]
reveal_type(AR + c16)  # E: Any

reveal_type(c8 + c16)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c8 + f8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c8 + i8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c8 + c8)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(c8 + f4)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(c8 + i4)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(c8 + b_)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(c8 + b)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(c8 + c)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c8 + f)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c8 + i)  # E: numpy.complexfloating[Any, Any]
reveal_type(c8 + AR)  # E: Any

reveal_type(c16 + c8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f8 + c8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(i8 + c8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(c8 + c8)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(f4 + c8)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(i4 + c8)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(b_ + c8)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(b + c8)  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(c + c8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f + c8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(i + c8)  # E: numpy.complexfloating[Any, Any]
reveal_type(AR + c8)  # E: Any

# Float

reveal_type(f8 + f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 + i8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 + f4)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 + i4)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 + b_)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 + b)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 + c)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f8 + f)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f8 + i)  # E: numpy.floating[Any]
reveal_type(f8 + AR)  # E: Any

reveal_type(f8 + f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i8 + f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f4 + f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i4 + f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b_ + f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(b + f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(c + f8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f + f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i + f8)  # E: numpy.floating[Any]
reveal_type(AR + f8)  # E: Any

reveal_type(f4 + f8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f4 + i8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f4 + f4)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(f4 + i4)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(f4 + b_)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(f4 + b)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(f4 + c)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f4 + f)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f4 + i)  # E: numpy.floating[Any]
reveal_type(f4 + AR)  # E: Any

reveal_type(f8 + f4)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i8 + f4)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(f4 + f4)  # E: umpy.floating[numpy.typing._32Bit]
reveal_type(i4 + f4)  # E: umpy.floating[numpy.typing._32Bit]
reveal_type(b_ + f4)  # E: umpy.floating[numpy.typing._32Bit]
reveal_type(b + f4)  # E: umpy.floating[numpy.typing._32Bit]
reveal_type(c + f4)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f + f4)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i + f4)  # E: numpy.floating[Any]
reveal_type(AR + f4)  # E: Any

# Int

reveal_type(i8 + i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 + u8)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(i8 + i4)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 + u4)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(i8 + b_)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 + b)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i8 + c)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(i8 + f)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i8 + i)  # E: numpy.signedinteger[Any]
reveal_type(i8 + AR)  # E: Any

reveal_type(u8 + u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 + i4)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(u8 + u4)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 + b_)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 + b)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u8 + c)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(u8 + f)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(u8 + i)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(u8 + AR)  # E: Any

reveal_type(i8 + i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(u8 + i8)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(i4 + i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(u4 + i8)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(b_ + i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(b + i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(c + i8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f + i8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i + i8)  # E: numpy.signedinteger[Any]
reveal_type(AR + i8)  # E: Any

reveal_type(u8 + u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(i4 + u8)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(u4 + u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(b_ + u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(b + u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(c + u8)  # E: numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]
reveal_type(f + u8)  # E: numpy.floating[numpy.typing._64Bit]
reveal_type(i + u8)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(AR + u8)  # E: Any

reveal_type(i4 + i8)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i4 + i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(i4 + i)  # E: numpy.signedinteger[Any]
reveal_type(i4 + b_)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(i4 + b)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(i4 + AR)  # E: Any

reveal_type(u4 + i8)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(u4 + i4)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(u4 + u8)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u4 + u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(u4 + i)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(u4 + b_)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(u4 + b)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(u4 + AR)  # E: Any

reveal_type(i8 + i4)  # E: numpy.signedinteger[numpy.typing._64Bit]
reveal_type(i4 + i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(i + i4)  # E: numpy.signedinteger[Any]
reveal_type(b_ + i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(b + i4)  # E: numpy.signedinteger[numpy.typing._32Bit]
reveal_type(AR + i4)  # E: Any

reveal_type(i8 + u4)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(i4 + u4)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(u8 + u4)  # E: numpy.unsignedinteger[numpy.typing._64Bit]
reveal_type(u4 + u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(b_ + u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(b + u4)  # E: numpy.unsignedinteger[numpy.typing._32Bit]
reveal_type(i + u4)  # E: Union[numpy.signedinteger[Any], numpy.floating[numpy.typing._64Bit]]
reveal_type(AR + u4)  # E: Any
