import pickle

from gym import error
from gym.utils import seeding


def test_invalid_seeds():
    for seed in [-1, "test"]:
        try:
            seeding.np_random(seed)
        except error.Error:
            pass
        else:
            assert False, f"Invalid seed {seed} passed validation"


def test_valid_seeds():
    for seed in [0, 1]:
        random, seed1 = seeding.np_random(seed)
        assert seed == seed1


def test_rng_pickle():
    rng, _ = seeding.np_random(seed=0)
    pickled = pickle.dumps(rng)
    rng2 = pickle.loads(pickled)
    assert isinstance(
        rng2, seeding.RandomNumberGenerator
    ), "Unpickled object is not a RandomNumberGenerator"
    assert rng.random() == rng2.random()
