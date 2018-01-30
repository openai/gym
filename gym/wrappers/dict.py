import gym
import numpy as np


__all__ = ['DictWrapper']


class DictWrapper(gym.ObservationWrapper):
    """Flattens selected keys of a Dict observation space into
    an array.
    """
    def __init__(self, env, dict_keys):
        super(DictWrapper, self).__init__(env)
        self.dict_keys = dict_keys

    def observation(self, observation):
        assert type(observation) == dict
        obs = []
        for key in self.dict_keys:
            obs.append(observation[key].flatten())
        return np.concatenate(obs)
