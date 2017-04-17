from gym.spaces import Discrete

def test_is_iterable():
    space = Discrete(5)
    assert len(space) == 5
    assert list(space) == [0, 1, 2, 3, 4]
