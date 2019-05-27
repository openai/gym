import numpy as np

from gym import RewardWrapper


class SignReward(RewardWrapper):
    r""""Bin reward to {-1, 0, +1} by its sign. """   
    def reward(self, reward):
        return np.sign(reward)
