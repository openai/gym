import numpy as np

import gym
from gym.spaces import prng, Discrete, Box
from gym.error import Error

class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)

    Note: A value of 0 always need to represent the NOOP action.

    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:

        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

    - Can be initialized as

        MultiDiscrete([ [0,4], [0,1], [0,1] ])

    """
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = prng.np_random.rand(self.num_discrete_space)
        return [int(x) for x in np.rint(np.multiply((self.high - self.low), random_array) + self.low)]
    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space
    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)
    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


# Adapters

class DiscreteToMultiDiscrete(Discrete):
    """
    Adapter that adapts the MultiDiscrete action space to a Discrete action space of any size

    The converted action can be retrieved by calling the adapter with the discrete action

        discrete_to_multi_discrete = DiscreteToMultiDiscrete(multi_discrete)
        discrete_action = discrete_to_multi_discrete.sample()
        multi_discrete_action = discrete_to_multi_discrete(discrete_action)

    It can be initialized using 3 configurations:

    Configuration 1) - DiscreteToMultiDiscrete(multi_discrete)                   [2nd param is empty]

        Would adapt to a Discrete action space of size (1 + nb of discrete in MultiDiscrete)
        where
            0   returns NOOP                                [  0,   0,   0, ...]
            1   returns max for the first discrete space    [max,   0,   0, ...]
            2   returns max for the second discrete space   [  0, max,   0, ...]
            etc.

    Configuration 2) - DiscreteToMultiDiscrete(multi_discrete, list_of_discrete) [2nd param is a list]

        Would adapt to a Discrete action space of size (1 + nb of items in list_of_discrete)
        e.g.
        if list_of_discrete = [0, 2]
            0   returns NOOP                                [  0,   0,   0, ...]
            1   returns max for first discrete in list      [max,   0,   0, ...]
            2   returns max for second discrete in list     [  0,   0,  max, ...]
            etc.

    Configuration 3) - DiscreteToMultiDiscrete(multi_discrete, discrete_mapping) [2nd param is a dict]

        Would adapt to a Discrete action space of size (nb_keys in discrete_mapping)
        where discrete_mapping is a dictionnary in the format { discrete_key: multi_discrete_mapping }

        e.g. for the Nintendo Game Controller [ [0,4], [0,1], [0,1] ] a possible mapping might be;

        mapping = {
            0:  [0, 0, 0],  # NOOP
            1:  [1, 0, 0],  # Up
            2:  [3, 0, 0],  # Down
            3:  [2, 0, 0],  # Right
            4:  [2, 1, 0],  # Right + A
            5:  [2, 0, 1],  # Right + B
            6:  [2, 1, 1],  # Right + A + B
            7:  [4, 0, 0],  # Left
            8:  [4, 1, 0],  # Left + A
            9:  [4, 0, 1],  # Left + B
            10: [4, 1, 1],  # Left + A + B
            11: [0, 1, 0],  # A only
            12: [0, 0, 1],  # B only,
            13: [0, 1, 1],  # A + B
        }

    """
    def __init__(self, multi_discrete, options=None):
        assert isinstance(multi_discrete, MultiDiscrete)
        self.multi_discrete = multi_discrete
        self.num_discrete_space = self.multi_discrete.num_discrete_space

        # Config 1
        if options is None:
            self.n = self.num_discrete_space + 1                # +1 for NOOP at beginning
            self.mapping = {i: [0] * self.num_discrete_space for i in range(self.n)}
            for i in range(self.num_discrete_space):
                self.mapping[i + 1][i] = self.multi_discrete.high[i]

        # Config 2
        elif isinstance(options, list):
            assert len(options) <= self.num_discrete_space
            self.n = len(options) + 1                          # +1 for NOOP at beginning
            self.mapping = {i: [0] * self.num_discrete_space for i in range(self.n)}
            for i, disc_num in enumerate(options):
                assert disc_num < self.num_discrete_space
                self.mapping[i + 1][disc_num] = self.multi_discrete.high[disc_num]

        # Config 3
        elif isinstance(options, dict):
            self.n = len(options.keys())
            self.mapping = options
            for i, key in enumerate(options.keys()):
                if i != key:
                    raise Error('DiscreteToMultiDiscrete must contain ordered keys. ' \
                                'Item {0} should have a key of "{0}", but key "{1}" found instead.'.format(i, key))
                if not self.multi_discrete.contains(options[key]):
                    raise Error('DiscreteToMultiDiscrete mapping for key {0} is ' \
                                'not contained in the underlying MultiDiscrete action space. ' \
                                'Invalid mapping: {1}'.format(key, options[key]))
        # Unknown parameter provided
        else:
            raise Error('DiscreteToMultiDiscrete - Invalid parameter provided.')

    def __call__(self, discrete_action):
        return self.mapping[discrete_action]


class BoxToMultiDiscrete(Box):
    """
    Adapter that adapts the MultiDiscrete action space to a Box action space

    The converted action can be retrieved by calling the adapter with the box action

        box_to_multi_discrete = BoxToMultiDiscrete(multi_discrete)
        box_action = box_to_multi_discrete.sample()
        multi_discrete_action = box_to_multi_discrete(box_action)

    It can be initialized using 2 configurations:

    Configuration 1) - BoxToMultiDiscrete(multi_discrete)                       [2nd param is empty]

        Would adapt to a Box action space of shape (nb of discrete space, ), with the min-max of
        each Discrete space sets as Box boundaries

        e.g. a MultiDiscrete with parameters [ [0,4], [0,1], [0,1] ], adapted through BoxToMultiDiscrete(multi_discrete)
            would adapt to a Box with parameters low=np.array([0.0, 0.0, 0.0]) high=np.array([4.0, 1.0, 1.0])

        The box action would then be rounded to the nearest integer.

        e.g. [ 2.560453, 0.3523456, 0.674546 ] would be converted to the multi discrete action of [3, 0, 1]

    Configuration 2) - BoxToMultiDiscrete(multi_discrete, list_of_discrete)     [2nd param is a list]

        Would adapt to a Box action space of shape (nb of items in list_of_discrete, ), where list_of_discrete
        is the index of the discrete space in the MultiDiscrete space

        e.g. a MultiDiscrete with parameters [ [0,4], [0,1], [0,1] ], adapted through BoxToMultiDiscrete(multi_discrete, [2, 0])
            would adapt to a Box with parameters low=np.array([0.0, 0.0]) high=np.array([1.0, 4.0])
            where
                0.0 = min(discrete space #2), 1.0 = max(discrete space #2)
                0.0 = min(discrete space #0), 4.0 = max(discrete space #0)

        The box action would then be rounded to the nearest integer and mapped to the correct discrete space in multi-discrete.

        e.g. [ 0.7412057, 3.0174142 ] would be converted to the multi discrete action of [3, 0, 1]

        This configuration is useful if you want to ignore certain discrete spaces in the MultiDiscrete space.

    """
    def __init__(self, multi_discrete, options=None):
        assert isinstance(multi_discrete, MultiDiscrete)
        self.multi_discrete = multi_discrete
        self.num_discrete_space = self.multi_discrete.num_discrete_space

        if options is None:
            options = list(range(self.num_discrete_space))

        if not isinstance(options, list):
            raise Error('BoxToMultiDiscrete - Invalid parameter provided.')

        assert len(options) <= self.num_discrete_space
        self.low = np.array([self.multi_discrete.low[x] for x in options])
        self.high = np.array([self.multi_discrete.high[x] for x in options])
        self.mapping = { i: disc_num for i, disc_num in enumerate(options)}

    def __call__(self, box_action):
        multi_discrete_action = [0] * self.num_discrete_space
        for i in self.mapping:
            multi_discrete_action[self.mapping[i]] = int(round(box_action[i], 0))
        return multi_discrete_action
