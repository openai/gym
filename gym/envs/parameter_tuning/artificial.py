import gym
import random
from gym import spaces
from six import StringIO
import sys
import numpy as np
from sklearn.svm import SVR

class TransferArtificial(gym.Env):
    """Environment for knowledge transfer in black box optimization. 
    Here training dataset is completely artificial.    
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, natural=False):
        """
        Initialize environment
        """
        self.action_space = spaces.Tuple((
                                          spaces.Box(-2.0,3.0, (1,1)), # C
                                          spaces.Box(-2.0,1.0, (1,1)), # gamma
                                           spaces.Discrete(2) # kernel type
                                           ))
        
        self.observation_space = spaces.Box(-1.0,1.0, (1,1))
        
        

        # Start the first game
        self._reset()

    def _step(self, action):
        """
        Perform some action in the environment
        """
        assert(self.action_space.contains(action))
        
        C, gamma, ktype = action;
        
        C = C[0]
        gamma = gamma[0]
        
        k = "linear"
        if ktype == 1:
            k = "rbf"
        
        model = SVR(C = 10**C, gamma=10**gamma, kernel=k)
        
        model.fit(self.X, self.Y)
        
        self.previous_acc = model.score(self.Xv, self.Yv)
        
        if self.previous_acc > self.best_val:
            self.best_val = self.previous_acc
        
        self.idx = self.idx + 1
        done = self.idx >= 20;
            
        return self._get_obs(), self.best_val, done, {}

    def _render(self, mode="human", close=False):
        
        if close:
            return
        
        print "Model #", self.idx, "best validation", self.best_val

    def _get_obs(self):
        """
        Observe the environment. Is usually used after the step is taken
        """
        # observation as per observation space 
        return np.array([[ self.previous_acc ]])

    def _reset(self):
        N, M = 100,5
        
        self.X = np.random.randn(N,M)
        self.Xv = np.random.randn(N,M)
        
        w = np.random.randn(M)
        
        def true_dep(X):
            return np.sin( np.dot( X, w ) )
        
        self.Y = true_dep(self.X)
        self.Yv = true_dep(self.Xv)
        
        self.idx = 0
        self.best_val = 0.0
        self.previous_acc = 0.0
        
        return self._get_obs()
