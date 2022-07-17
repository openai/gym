import pytest

from gym.spaces import Discrete, MultiBinary, Tuple


def test_tuple():
    spaces = [Discrete(5), Discrete(10), Discrete(5)]
    space_tuple = Tuple(spaces)

    assert len(space_tuple) == len(spaces)
    for i, space in enumerate(space_tuple):
        assert space == spaces[i]
    for i, space in enumerate(reversed(space_tuple)):
        assert space == spaces[len(spaces) - 1 - i]

    # Check Sequence attributes
    assert space_tuple.count(Discrete(5)) == 2
    assert space_tuple.count(MultiBinary(2)) == 0

    assert space_tuple.index(Discrete(5)) == 0
    assert space_tuple.index(Discrete(5), 1) == 2

    with pytest.raises(ValueError):
        space_tuple.index(Discrete(10), 0, 1)
    with pytest.raises(IndexError):
        assert space_tuple[4]  # To ensure that the statement has an effect
