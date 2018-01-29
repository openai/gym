import gym

__all__ = ['DictWrapper']

def DictWrapper(dict_keys):
    class DictWrapper(gym.ObservationWrapper):
        """
            Generic common frame skipping wrapper
            Will perform action for `x` additional steps
        """
        def __init__(self, env):
            super(DictWrapper, self).__init__(env)
            self.dict_keys = dict_keys

        def observation(self, observation):
            assert type(observation) == dict
            obs = []
            for key in self.dict_keys:
                obs.append(observation[key].flatten())
            return np.concatenate(obs)

    return DictWrapper
