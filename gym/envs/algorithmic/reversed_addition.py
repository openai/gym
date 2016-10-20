import numpy as np
from gym.envs.algorithmic import algorithmic_env

class ReversedAdditionEnv(algorithmic_env.GridAlgorithmicEnv):
    def __init__(self, rows=2, base=3):
        super(ReversedAdditionEnv, self).__init__(rows=rows, base=base, chars=False)

    def target_from_input_data(self, input_strings):
        curry = 0
        target = []
        for digits in input_strings:
            total = sum(digits) + curry
            target.append(total % self.base)
            curry = total / self.base

        if curry > 0:
            target.append(curry)
        return target
