import numpy as np
from gym.spaces.box import Box

np.set_printoptions(suppress=True)

space = Box(low=np.array([-np.inf, 0]), high=np.array([0.0, np.inf]), dtype=np.int64)

foo = np.array([0.0, np.iinfo(np.int64).max - 2], dtype=np.int64)
print(foo)
print(foo.dtype)
foo = foo.astype(np.int64)
print(foo)


bar = np.array([0.0, np.iinfo(np.int32).max - 2], dtype=np.int32)
print(bar)
print(bar.dtype)
bar = bar.astype(np.int64)
print(bar)
