import random
import numpy as np
from gym.envs.algorithmic import algorithmic_env
from gym.envs.algorithmic.algorithmic_env import ha

class ReversedAdditionEnv(algorithmic_env.AlgorithmicEnv):
    def __init__(self, rows=2, base=3):
        self.rows = rows
        algorithmic_env.AlgorithmicEnv.__init__(self,
                                                inp_dim=2,
                                                base=base,
                                                chars=False)
    def set_data(self):
        self.content = {}
        self.target = {}
        curry = 0
        for i in range(self.total_len):
            vals = []
            for k in range(self.rows):
                val = random.randrange(self.base)
                self.content[ha(np.array([i, k]))] = val
                vals.append(val)
            total = sum(vals) + curry
            self.target[i] = total % self.base
            curry = total / self.base
        if curry > 0:
            self.target[self.total_len] = curry
        self.total_reward = self.total_len 


