from gym import error
from gym.utils import seeding


def test_invalid_seeds():
    for seed in [-1, "test"]:
        try:
            seeding.np_random(seed)
        except error.Error:
            pass
        else:
            assert False, "Invalid seed {} passed validation".format(seed)


def test_valid_seeds():
    for seed in [0, 1]:
        random, seed1 = seeding.np_random(seed)
        assert seed == seed1
