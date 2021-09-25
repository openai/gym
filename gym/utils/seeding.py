import numpy as np

from gym import error


def np_random(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error(
            "Seed must be a non-negative integer or omitted, not {}".format(seed)
        )

    rng = np.random.default_rng(seed)
    seed = rng.bit_generator._seed_seq.entropy
    return rng, seed
