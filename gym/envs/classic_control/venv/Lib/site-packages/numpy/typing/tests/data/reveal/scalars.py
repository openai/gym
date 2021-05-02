import numpy as np

x = np.complex64(3 + 2j)

reveal_type(x.real)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(x.imag)  # E: numpy.floating[numpy.typing._32Bit]

reveal_type(x.real.real)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(x.real.imag)  # E: numpy.floating[numpy.typing._32Bit]

reveal_type(x.itemsize)  # E: int
reveal_type(x.shape)  # E: Tuple[]
reveal_type(x.strides)  # E: Tuple[]

reveal_type(x.ndim)  # E: Literal[0]
reveal_type(x.size)  # E: Literal[1]

reveal_type(x.squeeze())  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(x.byteswap())  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(x.transpose())  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]

reveal_type(x.dtype)  # E: numpy.dtype[numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]]

reveal_type(np.complex64().real)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(np.complex128().imag)  # E: numpy.floating[numpy.typing._64Bit]

reveal_type(np.unicode_('foo'))  # E: numpy.str_
reveal_type(np.str0('foo'))  # E: numpy.str_
