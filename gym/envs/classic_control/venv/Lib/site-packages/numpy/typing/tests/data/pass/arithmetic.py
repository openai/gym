import numpy as np

c16 = np.complex128(1)
f8 = np.float64(1)
i8 = np.int64(1)
u8 = np.uint64(1)

c8 = np.complex64(1)
f4 = np.float32(1)
i4 = np.int32(1)
u4 = np.uint32(1)

dt = np.datetime64(1, "D")
td = np.timedelta64(1, "D")

b_ = np.bool_(1)

b = bool(1)
c = complex(1)
f = float(1)
i = int(1)

AR = np.ones(1, dtype=np.float64)
AR.setflags(write=False)

# unary ops

-c16
-c8
-f8
-f4
-i8
-i4
-u8
-u4
-td
-AR

+c16
+c8
+f8
+f4
+i8
+i4
+u8
+u4
+td
+AR

abs(c16)
abs(c8)
abs(f8)
abs(f4)
abs(i8)
abs(i4)
abs(u8)
abs(u4)
abs(td)
abs(b_)
abs(AR)

# Time structures

dt + td
dt + i
dt + i4
dt + i8
dt - dt
dt - i
dt - i4
dt - i8

td + td
td + i
td + i4
td + i8
td - td
td - i
td - i4
td - i8
td / f
td / f4
td / f8
td / td
td // td
td % td


# boolean

b_ / b
b_ / b_
b_ / i
b_ / i8
b_ / i4
b_ / u8
b_ / u4
b_ / f
b_ / f8
b_ / f4
b_ / c
b_ / c16
b_ / c8

b / b_
b_ / b_
i / b_
i8 / b_
i4 / b_
u8 / b_
u4 / b_
f / b_
f8 / b_
f4 / b_
c / b_
c16 / b_
c8 / b_

# Complex

c16 + c16
c16 + f8
c16 + i8
c16 + c8
c16 + f4
c16 + i4
c16 + b_
c16 + b
c16 + c
c16 + f
c16 + i
c16 + AR

c16 + c16
f8 + c16
i8 + c16
c8 + c16
f4 + c16
i4 + c16
b_ + c16
b + c16
c + c16
f + c16
i + c16
AR + c16

c8 + c16
c8 + f8
c8 + i8
c8 + c8
c8 + f4
c8 + i4
c8 + b_
c8 + b
c8 + c
c8 + f
c8 + i
c8 + AR

c16 + c8
f8 + c8
i8 + c8
c8 + c8
f4 + c8
i4 + c8
b_ + c8
b + c8
c + c8
f + c8
i + c8
AR + c8

# Float

f8 + f8
f8 + i8
f8 + f4
f8 + i4
f8 + b_
f8 + b
f8 + c
f8 + f
f8 + i
f8 + AR

f8 + f8
i8 + f8
f4 + f8
i4 + f8
b_ + f8
b + f8
c + f8
f + f8
i + f8
AR + f8

f4 + f8
f4 + i8
f4 + f4
f4 + i4
f4 + b_
f4 + b
f4 + c
f4 + f
f4 + i
f4 + AR

f8 + f4
i8 + f4
f4 + f4
i4 + f4
b_ + f4
b + f4
c + f4
f + f4
i + f4
AR + f4

# Int

i8 + i8
i8 + u8
i8 + i4
i8 + u4
i8 + b_
i8 + b
i8 + c
i8 + f
i8 + i
i8 + AR

u8 + u8
u8 + i4
u8 + u4
u8 + b_
u8 + b
u8 + c
u8 + f
u8 + i
u8 + AR

i8 + i8
u8 + i8
i4 + i8
u4 + i8
b_ + i8
b + i8
c + i8
f + i8
i + i8
AR + i8

u8 + u8
i4 + u8
u4 + u8
b_ + u8
b + u8
c + u8
f + u8
i + u8
AR + u8

i4 + i8
i4 + i4
i4 + i
i4 + b_
i4 + b
i4 + AR

u4 + i8
u4 + i4
u4 + u8
u4 + u4
u4 + i
u4 + b_
u4 + b
u4 + AR

i8 + i4
i4 + i4
i + i4
b_ + i4
b + i4
AR + i4

i8 + u4
i4 + u4
u8 + u4
u4 + u4
b_ + u4
b + u4
i + u4
AR + u4
