import pytest

from gym.spaces import Box, Dict, Discrete, MultiBinary, Tuple


def test_sequence_inheritance():
    """The gym Tuple space inherits from abc.Sequences, this test checks all functions work"""
    spaces = [Discrete(5), Discrete(10), Discrete(5)]
    tuple_space = Tuple(spaces)

    assert len(tuple_space) == len(spaces)
    # Test indexing
    for i in range(len(tuple_space)):
        assert tuple_space[i] == spaces[i]

    # Test iterable
    for space in tuple_space:
        assert space in spaces

    # Test count
    assert tuple_space.count(Discrete(5)) == 2
    assert tuple_space.count(MultiBinary(2)) == 0

    # Test index
    assert tuple_space.index(Discrete(5)) == 0
    assert tuple_space.index(Discrete(5), 1) == 2

    # Test errors
    with pytest.raises(ValueError):
        tuple_space.index(Discrete(10), 0, 1)
    with pytest.raises(IndexError):
        assert tuple_space[4]


@pytest.mark.parametrize(
    "space, seed, expected_len",
    [
        (Tuple([Discrete(5), Discrete(4)]), None, 2),
        (Tuple([Discrete(5), Discrete(4)]), 123, 3),
        (Tuple([Discrete(5), Discrete(4)]), (123, 456), 2),
        (
            Tuple(
                (Discrete(5), Tuple((Box(low=0.0, high=1.0, shape=(3,)), Discrete(2))))
            ),
            (123, (456, 789)),
            3,
        ),
        (
            Tuple(
                (
                    Discrete(3),
                    Dict(position=Box(low=0.0, high=1.0), velocity=Discrete(2)),
                )
            ),
            (123, {"position": 456, "velocity": 789}),
            3,
        ),
    ],
)
def test_seeds(space, seed, expected_len):
    seeds = space.seed(seed)
    assert isinstance(seeds, list) and all(isinstance(elem, int) for elem in seeds)
    assert len(seeds) == expected_len
