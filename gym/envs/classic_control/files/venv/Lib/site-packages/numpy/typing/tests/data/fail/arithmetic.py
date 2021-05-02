import numpy as np

b_ = np.bool_()
dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

b_ - b_  # E: No overload variant

dt + dt  # E: Unsupported operand types
td - dt  # E: Unsupported operand types
td % 1  # E: Unsupported operand types
td / dt  # E: No overload
td % dt  # E: Unsupported operand types

-b_  # E: Unsupported operand type
+b_  # E: Unsupported operand type
