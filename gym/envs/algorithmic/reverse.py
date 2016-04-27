"""
Task is to reverse content over the input tape.
http://arxiv.org/abs/1511.07275
"""

import random
import numpy as np
from gym.envs.algorithmic import algorithmic_env
from gym.envs.algorithmic.algorithmic_env import ha

class ReverseEnv(algorithmic_env.AlgorithmicEnv):
    def __init__(self, base=2):
        algorithmic_env.AlgorithmicEnv.__init__(self,
                                                inp_dim=1,
                                                base=base,
                                                chars=True)
        algorithmic_env.AlgorithmicEnv.current_length = 1
        self.last = 50

    def set_data(self):
        self.content = {}
        self.target = {}
        for i in range(self.total_len):
            val = random.randrange(self.base)
            self.content[ha(np.array([i]))] = val
            self.target[self.total_len - i - 1] = val
        self.total_reward = self.total_len + 0.9
