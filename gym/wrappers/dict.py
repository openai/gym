import gym
import numpy as np


__all__ = ['FlattenDictWrapper']


class FlattenDictWrapper(gym.ObservationWrapper):
    """Flattens selected keys of a Dict observation space into
    an array.
    """
    def __init__(self, env, dict_keys):
        super(FlattenDictWrapper, self).__init__(env)
        self.dict_keys = dict_keys
        self.dtype = None

        # Figure out observation_space dimension.
        size = self.get_dict_size(self.env.observation_space.spaces, dict_keys)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype='float32')

    def get_dict_size(self, spaces, dict_keys):
        size = 0
        for key in dict_keys:
            size += self._size(spaces[key])
        return size

    def get_tuple_size(self, spaces):
        size = 0
        for space in spaces:
            size += self._size(space)
        return size

    def get_box_size(self, space):
        assert isinstance(space, gym.spaces.Box), "Only spaces of type Box are supported."
        if self.dtype is not None:
            assert space.dtype == self.dtype, "All spaces must have the same dtype."
        else:
            self.dtype = space.dtype
        shape = space.shape
        return np.prod(shape, dtype=np.int64)

    def _size(self, space):
        if isinstance(space, gym.spaces.Dict):
            return self.get_dict_size(space.spaces, space.spaces.keys())
        elif isinstance(space, gym.spaces.Tuple):
            return self.get_tuple_size(space.spaces)
        else:
            return self.get_box_size(space)

    def observation(self, observation):
        assert isinstance(observation, dict)
        return self.ravel_dict_observation(observation, self.dict_keys)

    def ravel_dict_observation(self, observation, dict_keys):
        assert isinstance(observation, dict)
        obs = []
        for key in dict_keys:
            obs.append(self._ravel(observation[key]))
        return np.concatenate(obs)

    def ravel_tuple_observation(self, observation):
        obs = []
        for item in observation:
            obs.append(self._ravel(item))
        return np.concatenate(obs)

    def _ravel(self, space):
        if isinstance(space, dict):
            return self.ravel_dict_observation(space, space.keys())
        elif isinstance(space, tuple):
            return self.ravel_tuple_observation(space)
        else:
            return np.array(space).ravel()

