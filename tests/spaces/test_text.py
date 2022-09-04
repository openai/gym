import re

import numpy as np
import pytest

from gym.spaces import Text


def test_sample_mask():
    space = Text(min_length=1, max_length=5)

    # Test the sample length
    sample = space.sample(mask=(3, None))
    assert sample in space
    assert len(sample) == 3

    sample = space.sample(mask=None)
    assert sample in space
    assert 1 <= len(sample) <= 5

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Trying to sample with a minimum length > 0 (1) but the character mask is all zero meaning that no character could be sampled."
        ),
    ):
        space.sample(mask=(3, np.zeros(len(space.character_set), dtype=np.int8)))

    space = Text(min_length=0, max_length=5)
    sample = space.sample(
        mask=(None, np.zeros(len(space.character_set), dtype=np.int8))
    )
    assert sample in space
    assert sample == ""

    # Test the sample characters
    space = Text(max_length=5, charset="abcd")

    sample = space.sample(mask=(3, np.array([0, 1, 0, 0], dtype=np.int8)))
    assert sample in space
    assert sample == "bbb"
