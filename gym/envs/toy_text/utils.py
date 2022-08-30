import numpy as np


def categorical_sample(prob_n, np_random: np.random.Generator):
    """Sample from categorical distribution where each row specifies class probabilities."""
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return np.argmax(csprob_n > np_random.random())
