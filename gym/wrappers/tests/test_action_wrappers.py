import gym
from gym import error
from gym.wrappers import action_wrappers as awrap
from gym import spaces
import numpy as np
import itertools

# discretize

def test_discretize_1d_box():
    cont = spaces.Box(np.array([0.0]), np.array([1.0]))
    disc, f = awrap.discretize(cont, 10)

    assert disc == spaces.Discrete(10)
    assert f(0) == 0.0
    assert f(9) == 1.0

def test_discretize_discrete():
    start = spaces.Discrete(5)
    d, f = awrap.discretize(start, 10)
    assert d == start

def test_discretize_nd_box():
    cont = spaces.Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    disc, f = awrap.discretize(cont, 10)

    assert disc == spaces.MultiDiscrete([(0, 9), (0, 9)])
    assert (f((0, 0)) == [0.0, 1.0]).all()
    assert (f((9, 9)) == [1.0, 2.0]).all()

    disc, f = awrap.discretize(cont, (5, 10))

    assert disc == spaces.MultiDiscrete([(0, 4), (0, 9)])
    assert (f((0, 0)) == [0.0, 1.0]).all()
    assert (f((4, 9)) == [1.0, 2.0]).all()

# flatten
def test_flatten_single():
    start = spaces.Discrete(5)
    d, f = awrap.flatten(start)
    assert d == start

    start = spaces.Box(np.array([0.0]), np.array([1.0]))
    d, f = awrap.flatten(start)
    assert d == start

def test_flatten_discrete():
    md = spaces.MultiDiscrete([(0, 2), (0, 3)])
    d, f = awrap.flatten(md)

    assert d == spaces.Discrete(12)
    # check that we get all actions exactly once
    actions = []
    for (i, j) in itertools.product([0, 1, 2], [0, 1, 2, 3]):
        actions += [(i, j)]
    for i in range(0, 12):
        a = f(i)
        assert a in actions, (a, actions)
        actions = list(filter(lambda x: x != a, list(actions)))
    assert len(actions) == 0

    # same test for binary
    md = spaces.MultiBinary(3)
    d, f = awrap.flatten(md)

    assert d == spaces.Discrete(2**3)
    # check that we get all actions exactly once
    actions = []
    for (i, j, k) in itertools.product([0, 1], [0, 1], [0, 1]):
        actions += [(i, j, k)]
    for i in range(0, 8):
        a = f(i)
        assert a in actions, (a, actions)
        actions = list(filter(lambda x: x != a, actions))
    assert len(actions) == 0

def test_flatten_continuous():
    ct = spaces.Box(np.zeros((2,2)), np.ones((2, 2)))
    d, f = awrap.flatten(ct)

    assert d == spaces.Box(np.zeros(4), np.ones(4))
    assert (f([1, 2, 3, 4]) == [[1, 2], [3, 4]]).all()

# rescale
def test_rescale_discrete():
    for s in [spaces.Discrete(10), spaces.MultiDiscrete([(0, 2), (0, 3)]), spaces.MultiBinary(5)]:
        try:
            awrap.rescale(s, 1.0)
            assert False 
        except TypeError: pass

def test_rescale_box():
    s = spaces.Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    d, f = awrap.rescale(s, np.array([1.0, 0.0]), np.array([2.0, 1.0]))

    assert d == spaces.Box(np.array([1.0, 0.0]), np.array([2.0, 1.0]))
    assert (f([1.0, 0.0]) == [0.0, 1.0]).all()
    assert (f([2.0, 1.0]) == [1.0, 2.0]).all()



# wrappers
class ExpectEnv(gym.Env):
    def __init__(self):
        super(ExpectEnv, self).__init__()

    def _step(self, action):
        assert action == self.expectation, "{} != {}".format(action, self.expectation)

def test_discretized_wrapper():
    gym.envs.register(id='ExpectTest-v0',
        entry_point='test_action_wrappers:ExpectEnv'
    )
    expect = gym.make("ExpectTest-v0")
    cont = spaces.Box(np.array([0.0]), np.array([1.0]))
    expect.action_space = cont
    expect.expectation  = 0.5
    wrapper = awrap.DiscretizedActionWrapper(expect, 3)
    wrapper.step(1)

def test_flattened_wrapper():
    gym.envs.register(id='ExpectTest-v1',
        entry_point='test_action_wrappers:ExpectEnv'
    )
    expect = gym.make("ExpectTest-v1")
    md = spaces.MultiDiscrete([(0, 1), (0, 1)])
    expect.action_space = md
    expect.expectation  = (1, 1)
    wrapper = awrap.FlattenedActionWrapper(expect)
    wrapper.step(3)

def test_rescaled_wrapper():
    gym.envs.register(id='ExpectTest-v2',
        entry_point='test_action_wrappers:ExpectEnv'
    )
    expect = gym.make("ExpectTest-v2")
    bx = spaces.Box(np.array([0.0]), np.array([1.0]))
    expect.action_space = bx
    expect.expectation  = 0.5
    wrapper = awrap.RescaledActionWrapper(expect, np.array([1.0]), np.array([2.0]))
    wrapper.step(1.5)


if __name__ == '__main__':
    test_discretize_1d_box()
    test_discretize_discrete()
    test_discretize_nd_box()

    test_flatten_single()
    test_flatten_discrete()
    test_flatten_continuous()

    test_rescale_discrete()
    test_rescale_box()

    test_discretized_wrapper()
    test_flattened_wrapper()
    test_rescaled_wrapper()
