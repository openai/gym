from six import StringIO
import sys
import numpy as np
import gym
from gym import spaces
from gym import error
from gym.utils import seeding


class GenericPOMDPEnv(gym.Env):
    def __init__(self, nb_base_states=None, nb_actions=None, confusion_dim=None, transition_table=None,
                 nb_unobservable=0, init_state=None, confusion_level=0.1, good_terminals=list(),
                 bad_terminals=list(), max_move=100, overwrite=False):
        if transition_table is not None:
            self.__dict__.update(locals())
        else:  # default POMDP
            self.nb_states = 10
            self.nb_actions = 4
            self.nb_unobservable = 0
            self.confusion_dim = self.nb_states
            self.confusion_level = confusion_level
            self.overwrite = overwrite
            self.good_terminals = [9]
            self.bad_terminals = [8]
            self.transition_table = np.array([
                [0, 0, 1],
                [0, 1, 2],
                [0, 2, 3],
                [1, 1, 4],
                [1, 2, 5],
                [2, 0, 4],
                [2, 2, 6],
                [3, 0, 5],
                [3, 1, 6],
                [4, 2, 7],
                [5, 1, 7],
                [6, 0, 7],
                [0, 3, 8],
                [1, 3, 8],
                [2, 3, 8],
                [3, 3, 8],
                [4, 3, 8],
                [5, 3, 8],
                [6, 3, 8],
                [7, 3, 9]], dtype='int32')
            self.init_state = 0
            self.max_move = max_move

    def _step(self, action):
        if self.done or self.move > self.max_move:
            self.done = True
            return self.obs, None, self.done, {'state': self.state, 'step': self.move}
        self.move += 1
        next_state = self.state  # self-loop if not in transition_table
        for t in self.transition_table:
            if t[0] == self.state and t[1] == action:
                next_state = t[2]
                break
        self.state = next_state
        self.obs = self.state2obs(next_state)
        if next_state in self.good_terminals:
            self.done = True
            reward = 1.
        elif next_state in self.bad_terminals:
            self.done = True
            reward = -1.
        elif self.move == self.max_move:
            self.done = True
            reward = -1.
        else:
            self.done = False
            reward = -1. / self.max_move
        return self.obs, reward, self.done, {'state': self.state, 'step': self.move}

    def _reset(self):
        self.move = 0
        self.state = self.init_state
        self.obs = self.state2obs(self.init_state)
        self.action_space = spaces.Discrete(self.nb_actions)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.nb_states - self.nb_unobservable + self.confusion_dim,))
        self.done = False
        return self.obs

    def _render(self, mode='std', close=False):
        if close:
            return
        output = StringIO() if mode == 'ansi' else sys.stdout
        output.write('base state: ', self.state)
        output.write('observation: ', self.obs)
        if mode == 'ansi':
            return output

    def state2obs(self, s_id):
        dim = self.confusion_dim + self.nb_states
        s = np.zeros(dim, dtype='float32')
        s[: self.confusion_dim] = np.random.uniform(-self.confusion_level, self.confusion_level,
                                                    size=self.confusion_dim)
        s[self.confusion_dim + s_id] = 1.

        # TODO: remove indices of unobservable states
        if self.nb_unobservable > 0:
            pass

        if not hasattr(self, "randproj"):
            # put confusion lazily
            self.randproj = np.eye(dim) - np.random.uniform(-self.confusion_level, self.confusion_level,
                                                            size=(dim, dim))
            self.invrandproj = np.linalg.inv(self.randproj)
            np.save('confusion.npy', (self.randproj, self.invrandproj))
        s = np.dot(self.randproj, s)
        return s

    def obs2state(self, s):
        s = np.dot(self.invrandproj, s)
        s1 = s[self.confusion_dim:]
        return np.argmax(s1)

    def write_mdp_to_dot(self, file='mdp.dot'):
        # after calling this method use the following for example:
        #    $ dot -T png -O mdp.dot
        import networkx as nx
        g = nx.DiGraph()
        g.add_nodes_from(np.arange(self.nb_states))
        edges = [(tr[0], tr[2], {'label': tr[1]}) for tr in self.transition_table]
        g.add_edges_from(edges)
        nx.drawing.nx_pydot.write_dot(g, file)
