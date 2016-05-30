import numpy as np

import gym
from gym.spaces import prng

class HighLow(gym.Space):
    """
    A matrix of dimensions n x 3, where

    - n is the number of options in the space (e.g. buttons that can be pressed simultaneously)
    - u[1] (the first column) is the minimum value (inclusive) that the option can have
    - u[2] (the second column) is the maximum value (inclusive) that the option can have
    - u[3] (the third column) is the precision (0 = rounded to integer, 2 = rounded to 2 decimals)

    e.g. if the space is composed of ATTACK (values: 0-100), MOVE_LEFT(0-1), MOVE_RIGHT(0,1)
    the space would be [ [0.0, 100.0, 2], [0, 1, 0], [0, 1, 0] ]
    """
    def __init__(self, matrix):
        """
        A matrix of shape (n, 3), where the first column is the minimum (inclusive), the second column
        is the maximum (inclusive), and the third column is the precision (number of decimals to keep)

        e.g. np.matrix([[0, 1, 0], [0, 1, 0], [0.0, 100.0, 2]])
        """
        (num_rows, num_cols) = matrix.shape
        assert num_rows >= 1
        assert num_cols == 3
        self.matrix = matrix
        self.num_rows = num_rows

    def sample(self):
        # For each row: round(random .* (max - min) + min, precision)
        max_minus_min = self.matrix[:, 1] - self.matrix[:, 0]
        random_matrix = np.multiply(max_minus_min, prng.np_random.rand(self.num_rows, 1)) + self.matrix[:, 0]
        rounded_matrix = np.zeros(self.num_rows)
        for i in range(self.num_rows):
            rounded_matrix[i] = round(random_matrix[i, 0], int(self.matrix[i, 2]))
        return rounded_matrix

    def contains(self, x):
        if x.shape[0] != self.num_rows:
            return False
        for i in range(self.num_rows):
            if not (self.matrix[i, 0] <= x[i] <= self.matrix[i, 1]):
                return False
        return True

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()
    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    @property
    def shape(self):
        return self.matrix.shape
    def __repr__(self):
        return "High-Low" + str(self.shape)
    def __eq__(self, other):
        return self.matrix == other.matrix
