import numpy as np

def rmse_func(ee_points):
    """
    Computes the Residual Mean Square Error of the difference between current and desired end-effector position
    """
    rmse = np.sqrt(np.mean(np.square(ee_points), dtype=np.float32))
    return rmse
