from gym.spaces import Tuple, Box, Discrete
import itertools
import numpy as np


def flatten_spaces(space):
    """
    Flatten a tree of Tuple spaces into a list of simple spaces.
    :param space: A space that may be a Tuple
    :return: a list of spaces
    """
    if isinstance(space, Tuple):
        return list(itertools.chain.from_iterable(flatten_spaces(s) for s in space.spaces))
    else:
        return [space]


def space_shapes(space_list):
    """
    Return shapes of a list of Discrete and Box spaces
    :param space_list: list of spaces
    :return: list of shape tuples
    """
    shapes = []
    for space in space_list:
        if isinstance(space, Discrete):
            shapes.append((space.n,))
        elif isinstance(space, Box):
            shapes.append(space.shape)
        else:
            raise NotImplementedError("Only Discrete and Box input spaces currently supported")
    return shapes


def one_hot(label, nb_classes):
    """
    Convert label and number of classes to one-hot representation
    :param label: class label
    :param nb_classes: number of classes
    :return: vector of length nb_classes, 1 at position label, 0 everywhere else.
    """
    ret = np.zeros((nb_classes,))
    ret[label] = 1
    return ret


def flatten_input(observation, observation_space):
    """
    Given observation and observation space, return list of numpy arrays of same shape as space_shapes.
    * Tuple spaces are flattened into a list
    * Box spaces are wrapped as numpy arrays and shaped to match the space.shape
    * Discrete spaces are one-hot encoded as numpy arrays
    :param observation: input observation, typically tuples of arrays
    :param observation_space: space describing the input observations
    :return: list of numpy arrays
    """
    if isinstance(observation_space, Tuple):
        return list(
            itertools.chain.from_iterable(flatten_input(o, s) for o, s in zip(observation, observation_space.spaces)))
    elif isinstance(observation_space, Discrete):
        return [one_hot(observation, observation_space.n)]
    elif isinstance(observation_space, Box):
        return [np.array(observation).reshape(observation_space.shape)]
    else:
        raise NotImplementedError(
            "Only Discrete and Box input spaces currently supported, found: {}".format(type(observation_space)))


def concatenated_input_dim(observation_space):
    """
    Calculate the total number of dimensions of an input space
    * Tuple spaces are unrolled
    * Boxe spaces are flattened
    * Discrete spaces are one-hot encoded
    :param observation_space: space describing the input observation
    :return: number of dimensions
    """
    shapes = space_shapes(flatten_spaces(observation_space))
    return np.sum(np.prod(shape) for shape in shapes)


def concatenated_input(observation, observation_space):
    """
    Convert Discrete spaces to one-hot encodings and flatten all inputs into a single numpy array.
    :param observation: input observation
    :param observation_space: space describing the input observation
    :return: single dimensional numpy array concatenating all input
    """
    observations = flatten_input(observation, observation_space)
    return np.hstack(obs.reshape((-1)) for obs in observations)
