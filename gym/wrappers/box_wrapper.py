import gym
from gym.spaces import *

import numpy as np

__all__ = ['BoxWrapper']

class FlattenConvertor:
    def __init__(self, space):
        assert(isinstance(space, Box))
        self.in_space = space
        self.out_space = Box(space.low.flatten(), space.high.flatten())
    
    def __call__(self, x):
        assert(self.in_space.contains(x))
        return x.flatten()

class OneHotConvertor:
    def __init__(self, space):
        assert(isinstance(space, Discrete))
        self.in_space = space
        self.out_space = Box(0, 1, [space.n])
    
    def __call__(self, x):
        assert(self.in_space.contains(x))
        a = np.zeros([self.in_space.n])
        a[x] = 1
        return a

class ConcatConvertor:
    def __init__(self, space):
        assert(isinstance(space, Tuple))
        
        self.in_space = space
        
        self.convertors = list(map(convertor, space.spaces))
        
        low = np.concatenate([c.out_space.low for c in self.convertors])
        high = np.concatenate([c.out_space.high for c in self.convertors])
        
        self.out_space = Box(low, high)
    
    def __call__(self, xs):
        #assert(self.in_space.contains(xs))
        return np.concatenate([c(x) for c, x in zip(self.convertors, xs)])

def convertor(space):
    if isinstance(space, Box):
        return FlattenConvertor(space)
    elif isinstance(space, Discrete):
        return OneHotConvertor(space)
    elif isinstance(space, Tuple):
        return ConcatConvertor(space)
    else:
        raise ValueError("Unsupported space %s" % space)

class BoxWrapper(gym.Wrapper):
    "Turns any observation space into a box."
    def __init__(self, env):
        super(BoxWrapper, self).__init__(env)
        self.convertor = convertor(env.observation_space)
        self.observation_space = self.convertor.out_space 

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.convertor(obs)
        return obs, reward, done, info
    
    def _reset(self):
        obs = self.env._reset()
        return self.convertor(obs)

