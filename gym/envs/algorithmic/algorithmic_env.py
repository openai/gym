from gym import Env
from gym.spaces import Discrete, Tuple
from gym.utils import colorize
import numpy as np
import random
from six import StringIO
import sys
import math

hash_base = None
def ha(array):
    return (hash_base * (array + 5)).sum()

class AlgorithmicEnv(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, inp_dim=1, base=10, chars=False):
        global hash_base
        hash_base = 50 ** np.arange(inp_dim)
        self.base = base
        self.last = 10
        self.total_reward = 0
        self.sum_reward = 0
        AlgorithmicEnv.sum_rewards = []
        self.chars = chars
        self.inp_dim = inp_dim
        AlgorithmicEnv.current_length = 2
        tape_control = []
        self.action_space = Tuple(([Discrete(2 * inp_dim), Discrete(2), Discrete(self.base)]))
        self.observation_space = Discrete(self.base + 1)
        self.reset()

    def _get_obs(self, pos=None):
        if pos is None:
            pos = self.x
        assert(isinstance(pos, np.ndarray) and pos.shape[0] == self.inp_dim)
        if ha(pos) not in self.content:
            self.content[ha(pos)] = self.base
        return self.content[ha(pos)]

    def _get_str_obs(self, pos=None):
        ret = self._get_obs(pos)
        if ret == self.base:
            return " "
        else:
            if self.chars:
                return chr(ret + ord('A'))
            return str(ret)

    def _get_str_target(self, pos=None):
        if pos not in self.target:
            return " "
        else:
            ret = self.target[pos]
            if self.chars:
                return chr(ret + ord('A'))
            return str(ret)

    def _render_observation(self):
        x = self.x
        if self.inp_dim == 1:
            x_str =      "Observation Tape    : "
            for i in range(-2, self.total_len + 2):
                if i == x:
                    x_str += colorize(self._get_str_obs(np.array([i])), 'green', highlight=True)
                else:
                    x_str += self._get_str_obs(np.array([i]))
            x_str += "\n"
            return x_str
        elif self.inp_dim == 2:
            label =      "Observation Grid    : "
            x_str = ""
            for j in range(-1, 3):
                if j != -1:
                    x_str += " " * len(label)
                for i in range(-2, self.total_len + 2):
                    if i == x[0] and j == x[1]:
                        x_str += colorize(self._get_str_obs(np.array([i, j])), 'green', highlight=True)
                    else:
                        x_str += self._get_str_obs(np.array([i, j]))
                x_str += "\n"
            x_str = label + x_str
            return x_str
        else:
            assert(False)


    def _render(self, mode='human', close=False):
        if close:
            # Nothing interesting to close
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        inp = "Total length of input instance: %d, step: %d\n" % (self.total_len, self.time)
        outfile.write(inp)
        x, y, action = self.x, self.y, self.last_action
        if action is not None:
            inp_act, out_act, pred = action
        outfile.write("=" * (len(inp) - 1) + "\n")
        y_str =      "Output Tape         : "
        target_str = "Targets             :   "
        if action is not None:
            if self.chars:
                pred_str = chr(pred + ord('A'))
            else:
                pred_str = str(pred)
        x_str = self._render_observation()
        max_len = int(self.total_reward) + 1
        for i in range(-2, max_len):
            if i not in self.target:
                y_str += " "
                continue
            target_str += self._get_str_target(i)
            if i < y - 1:
                y_str += self._get_str_target(i)
            elif i == (y - 1):
                if action is not None and out_act == 1:
                    if pred == self.target[i]:
                        y_str += colorize(pred_str, 'green', highlight=True)
                    else:
                        y_str += colorize(pred_str, 'red', highlight=True)
                else:
                    y_str += self._get_str_target(i)
        outfile.write(x_str)
        outfile.write(y_str + "\n")
        outfile.write(target_str + "\n\n")

        if action is not None:
            outfile.write("Current reward      :   %.3f\n" % self.reward)
            outfile.write("Cumulative reward   :   %.3f\n" % self.sum_reward)
            move = ""
            if inp_act == 0:
                move = "left"
            elif inp_act == 1:
                move = "right"
            elif inp_act == 2:
                move += "up"
            elif inp_act == 3:
                move += "down"
            outfile.write("Action              :   Tuple(move over input: %s,\n" % move)
            if out_act == 1:
                out_act = "True"
            else:
                out_act = "False"
            outfile.write("                              write to the output tape: %s,\n" % out_act)
            outfile.write("                              prediction: %s)\n" % pred_str)
        else:
            outfile.write("\n" * 5)
        return outfile

    def _step(self, action):
        self.last_action = action
        inp_act, out_act, pred = action
        done = False
        reward = 0.0
        # We are outside the sample.
        self.time += 1
        if self.y not in self.target:
            reward = -10.0
            done = True
        else:
            if out_act == 1:
                if pred == self.target[self.y]:
                    reward = 1.0
                else:
                    reward = -0.5
                    done = True
                self.y += 1
                if self.y not in self.target:
                    done = True
            if inp_act == 0:
                self.x[0] -= 1
            elif inp_act == 1:
                self.x[0] += 1
            elif inp_act == 2:
                self.x[1] -= 1
            elif inp_act == 3:
                self.x[1] += 1
            if self.time > self.total_len + self.total_reward + 4:
                reward = -1.0
                done = True
        obs = self._get_obs()
        self.reward = reward
        self.sum_reward += reward
        return (obs, reward, done, {})

    def _reset(self):
        self.last_action = None
        self.x = np.zeros(self.inp_dim).astype(np.int)
        self.y = 0
        AlgorithmicEnv.sum_rewards.append(self.sum_reward - self.total_reward)
        AlgorithmicEnv.sum_rewards = AlgorithmicEnv.sum_rewards[-self.last:]
        if len(AlgorithmicEnv.sum_rewards) == self.last and \
          min(AlgorithmicEnv.sum_rewards) >= -1.0 and \
          AlgorithmicEnv.current_length < 30:
            AlgorithmicEnv.current_length += 1
            AlgorithmicEnv.sum_rewards = []
        self.sum_reward = 0.0
        self.time = 0
        self.total_len = random.randrange(3) + AlgorithmicEnv.current_length
        self.set_data()
        return self._get_obs()
