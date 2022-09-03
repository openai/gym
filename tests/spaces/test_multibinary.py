import numpy as np

from gym.spaces import MultiBinary


def test_sample():
    space = MultiBinary(4)

    sample = space.sample(mask=np.array([0, 0, 1, 1], dtype=np.int8))
    assert np.all(sample == [0, 0, 1, 1])

    sample = space.sample(mask=np.array([0, 1, 2, 2], dtype=np.int8))
    assert sample[0] == 0 and sample[1] == 1
    assert sample[2] == 0 or sample[2] == 1
    assert sample[3] == 0 or sample[3] == 1

    space = MultiBinary(np.array([2, 3]))
    sample = space.sample(mask=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int8))
    assert np.all(sample == [[0, 0, 0], [1, 1, 1]]), sample
