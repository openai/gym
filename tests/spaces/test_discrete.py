import numpy as np

from gym.spaces import Discrete


def test_space_legacy_pickling():
    legacy_state = {
        "shape": (
            1,
            2,
            3,
        ),
        "dtype": np.int64,
        "np_random": np.random.default_rng(),
        "n": 3,
    }
    space = Discrete(1)
    space.__setstate__(legacy_state)

    assert space.shape == legacy_state["shape"]
    assert space.np_random == legacy_state["np_random"]
    assert space.n == 3
    assert space.dtype == legacy_state["dtype"]

    # Test that start is missing
    assert "start" in space.__dict__
    del space.__dict__["start"]  # legacy did not include start param
    assert "start" not in space.__dict__

    space.__setstate__(legacy_state)
    assert space.start == 0
