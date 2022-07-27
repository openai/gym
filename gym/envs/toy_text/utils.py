import numpy as np

from gym.utils import seeding


def categorical_sample(prob_n, np_random: seeding.RandomNumberGenerator):
    """Sample from categorical distribution where each row specifies class probabilities."""
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return np.argmax(csprob_n > np_random.random())
