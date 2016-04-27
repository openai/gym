"""
Task is to copy content multiple-times from the input tape to
the output tape. http://arxiv.org/abs/1511.07275
"""
import random
import numpy as np
from gym.envs.algorithmic import algorithmic_env
from gym.envs.algorithmic.algorithmic_env import ha

class RepeatCopyEnv(algorithmic_env.AlgorithmicEnv):
    def __init__(self, base=5):
        algorithmic_env.AlgorithmicEnv.__init__(self,
                                                inp_dim=1,
                                                base=base,
                                                chars=True)
        self.last = 50

    def set_data(self):
        self.content = {}
        self.target = {}
        unique = set()
        for i in range(self.total_len):
            val = random.randrange(self.base)
            self.content[ha(np.array([i]))] = val
            self.target[i] = val
            self.target[2 * self.total_len - i - 1] = val
            self.target[2 * self.total_len + i] = val
        self.total_reward = 3.0 * self.total_len + 0.9

