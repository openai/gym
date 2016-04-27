"""
Task is to return every second character from the input tape.
http://arxiv.org/abs/1511.07275
"""

import random
import numpy as np
from gym.envs.algorithmic import algorithmic_env
from gym.envs.algorithmic.algorithmic_env import ha

class DuplicatedInputEnv(algorithmic_env.AlgorithmicEnv):
    def __init__(self, duplication=2, base=5):
        self.duplication = duplication
        algorithmic_env.AlgorithmicEnv.__init__(self,
                                                inp_dim=1,
                                                base=base,
                                                chars=True)
    def set_data(self):
        self.content = {}
        self.target = {}
        copies = int(self.total_len / self.duplication)
        for i in range(copies):
            val = random.randrange(self.base)
            self.target[i] = val
            for d in range(self.duplication):
                self.content[ha(np.array([i * self.duplication + d]))] = val
        self.total_reward = self.total_len / self.duplication
