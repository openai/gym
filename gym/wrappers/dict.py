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
            space = spaces[key]
            if isinstance(space, gym.spaces.Dict):
                size += self.get_dict_size(space.spaces, space.spaces.keys())
            elif isinstance(space, gym.spaces.Tuple):
                size += self.get_tuple_size(space.spaces)
            else:
                size += self.get_box_size(space)
        return size

    def get_tuple_size(self, spaces):
        size = 0
        for space in spaces:
            if isinstance(space, gym.spaces.Dict):
                size += self.get_dict_size(space.spaces, space.spaces.keys())
            elif isinstance(space, gym.spaces.Tuple):
                size += self.get_tuple_size(space.spaces)
            else:
                size += self.get_box_size(space)
        return size

    def get_box_size(self, space):
        assert isinstance(space, gym.spaces.Box), "Only spaces of type Box are supported."
        if self.dtype is not None:
            assert space.dtype == self.dtype, "All spaces must have the same dtype."
        else:
            self.dtype = space.dtype
        shape = space.shape
        return np.prod(shape, dtype=np.int64)

    def observation(self, observation):
        assert isinstance(observation, dict)
        obs = []
        for key in self.dict_keys:
            obs.append(observation[key].ravel())
        return np.concatenate(obs)
