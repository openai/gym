from gym import spaces, ActionWrapper
import numpy as np
import itertools
import numbers

def is_discrete(space):
    # check if space is discrete
    if isinstance(space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
        return True
    elif isinstance(space, spaces.Box):
        return False
    raise TypeError("Unknown space {} supplied".format(type(space)))

def is_compound(space):
    if isinstance(space, spaces.Discrete):
        return False
    elif isinstance(space, spaces.Box):
        return len(space.shape) != 1 or space.shape[0] != 1
    elif isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        return True

    raise TypeError("Unknown space %r supplied"%type(space))


# Discretization 
def discretize(space, steps):
    # there are two possible ways how we could handle already
    # discrete spaces. 
    #  1) throw an error because (unless
    #     steps is configured to fit) we would try to convert 
    #     an already discrete space to one with a different number
    #     of states.
    #  2) keep the space as is.
    # here, we implement the second. This allows scripts that 
    # train a discrete agent to just apply discretize, only 
    # changing envs that are not already discrete.
    if is_discrete(space):
        return space, lambda x: x

    # check that step number is valid and convert steps into a np array
    if not isinstance(steps, numbers.Integral):
        steps = np.array(steps, dtype=int)
        if (steps < 2).any():
            raise ValueError("Need at least two steps to discretize, got {}".format(steps))
    elif steps < 2:
        raise ValueError("Need at least two steps to discretize, got %i"%steps)

    if isinstance(space, spaces.Box):
        if len(space.shape) == 1 and space.shape[0] == 1:
            discrete_space = spaces.Discrete(steps)
            lo = space.low[0]
            hi = space.high[0]
            def convert(x):
                return lo + (hi - lo) * float(x) / (steps-1)
            return discrete_space, convert
        else:
            if isinstance(steps, numbers.Integral):
                steps = np.full(space.low.shape, steps)
            assert steps.shape == space.shape, "supplied steps have invalid shape"
            starts = np.zeros_like(steps)
            # MultiDiscrete is inclusive, thus we need steps-1 as last value
            # currently, MultiDiscrete iterates twice over its input, which is not possible for a zip
            # result in python 3
            discrete_space = spaces.MultiDiscrete(list(zip(starts.flatten(), (steps-1).flatten())))
            lo = space.low.flatten()
            hi = space.high.flatten()
            def convert(x):
                return np.reshape(lo + (hi - lo) * x / (steps-1), space.shape)
            return discrete_space, convert
    raise ValueError()

# Flattening
def flatten(space):
    # no need to do anything if already flat
    if not is_compound(space):
        return space, lambda x: x

    if isinstance(space, spaces.Box):
        shape = space.low.shape
        lo = space.low.flatten()
        hi = space.high.flatten()
        def convert(x):
            return np.reshape(x, shape)
        return spaces.Box(low=lo, high=hi), convert

    elif isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        if isinstance(space, spaces.MultiDiscrete):
            ranges = [range(space.low[i], space.high[i]+1, 1) for i in range(space.num_discrete_space)]
        elif isinstance(space, spaces.MultiBinary):
            ranges = [range(0, 2) for i in range(space.n)]
        prod   = itertools.product(*ranges)
        lookup = list(prod)
        dspace = spaces.Discrete(len(lookup))
        convert = lambda x: lookup[x]
        return dspace, convert

    raise TypeError("Cannot flatten {}".format(type(space)))


# rescale a continuous action space
def rescale(space, low, high):
    if is_discrete(space):
        raise TypeError("Cannot rescale discrete space {}".format(space))

    if not isinstance(space, spaces.Box):
        raise TypeError("Cannot rescale non-Box space {}".format(space))

    lo = space.low
    hi = space.high
    rg = hi - lo
    rs = high - low
    sc = rg / rs
    def convert(x):
        y = (x - low) * sc # y is in [0, rg]
        return y + space.low
    return spaces.Box(low, high), convert


###########################################################################
class FlattenedActionWrapper(ActionWrapper):
    def __init__(self, env):
        super(FlattenedActionWrapper, self).__init__(env)
        s, a = flatten(env.action_space)
        self.action_space = s
        self._action = a

class DiscretizedActionWrapper(ActionWrapper):
    def __init__(self, env, steps):
        super(DiscretizedActionWrapper, self).__init__(env)
        s, a = discretize(env.action_space, steps)
        self.action_space = s
        self._action = a

class RescaledActionWrapper(ActionWrapper):
    def __init__(self, env, low, high):
        super(RescaledActionWrapper, self).__init__(env)
        s, a = rescale(env.action_space, low=low, high=high)
        self.action_space = s
        self._action = a
