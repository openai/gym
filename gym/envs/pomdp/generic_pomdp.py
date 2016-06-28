from six import StringIO
import sys
import os
import numpy as np
import gym
from gym import spaces
from gym import error
from gym.utils import seeding, colorize
import logging
logger = logging.getLogger(__name__)


class GenericPOMDPEnv(gym.Env):
    """
    A generic POMDP implementation.

    It supports an underlying MDP with additional clutter (random) state variables.
    The observables are then produced by multiplying the cluttered state vector with a confusion matrix:

        Obs = (I - Rand(square: size = #State + #Clutter)) * (State .concat. Clutter)

    The environment also supports two separate sets of "good" and "bad" states, entering to which will define rewards.

    The reward signal is next computed using the following scheme:
        +1.0  if entering a good terminal state
        -1.0  if entering a bad terminal state
        -1.0  if reaching max_move
        -1.0/max_move  otherwise

    Additionally, GenericPOMDPEnv supports partial observability. Any state in `unobservable_states` will be removed
    from above equation for computing Obs and the size of matrices will be set accordingly.

    Note: In order to have a MDP, simply use `confusion_level=0.0`, `clutter_dim=0`, and `unobservable_states=[]`.
    """
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, nb_states=None, nb_actions=None, clutter_dim=None, transition_table=None,
                 unobservable_states=list(), init_state=None, confusion_level=0.1, good_terminals=list(),
                 bad_terminals=list(), max_move=100, confusion_file='confusion.npy', overwrite=False):
        """
        Args:
            nb_states:  number of MDP states
            nb_actions: number of actions
            clutter_dim: number of clutter states
            transition_table: MDP transition table: list([s, a, s', p]) --> p = transition probability
            unobservable_states: unobservable MDP states or empty list if fully observable
            init_state: MDP init state
            confusion_level: in [0, 1], level of confusion --  0 == no confusion
            good_terminals: list of good terminal states resulting in reward +1
            bad_terminals: list of bad terminal states resulting in reward -1
            max_move: maximum allowable number of steps (terminal state if reached with reward -1)
            overwrite: boolean for overwriting saved confusion matrix
        """
        assert None not in (nb_states, nb_actions, clutter_dim, transition_table, init_state) and \
            len(good_terminals) > 0, 'Bad one or more input arguments.'
        self.__dict__.update(locals())
        self.unobservable_states = np.asarray(unobservable_states)
        self.nb_unobservable = len(unobservable_states)
        self.dim = self.clutter_dim + self.nb_states
        self.confusion = np.eye(self.dim) - np.random.uniform(-self.confusion_level,
                                                              self.confusion_level,
                                                              size=(self.dim, self.dim))
        if not os.path.isfile(self.confusion_file) or self.overwrite:
            np.save(self.confusion_file, self.confusion)
        else:
            logger.warning('File "{0}" exists. Set `overwrite=True` to permit overwrite.'.format(self.confusion_file))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _configure(self, confusion_matrix):
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1] == self.dim, \
            'Provided confusion matrix does not match observable dim {0}.'.format(self.dim)
        self.confusion = confusion_matrix

    def _step(self, action):
        if self.done or self.move > self.max_move:
            self.done = True
            return self.obs, None, self.done, {'state': self.state, 'step': self.move}
        self.move += 1

        t_list = []
        for t in self.transition_table:
            if t[0] == self.state and t[1] == action:
                t_list.append(t)
        if t_list:  # otherwise self-loop (if not in the transition_table)
            t_probs = [item[-1] for item in t_list]
            t_idx = np.random.multinomial(n=1, pvals=np.array(t_probs)).argmax()
            self.state = t_list[t_idx][2]

        self.obs = self.state2obs(self.state)
        if self.state in self.good_terminals:
            self.done = True
            reward = 1.
        elif self.state in self.bad_terminals:
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
                                            shape=(self.nb_states - self.nb_unobservable + self.clutter_dim,))
        self.done = False
        return self.obs

    def _render(self, mode='human', close=False):
        if close:
            return
        output = StringIO() if mode == 'ansi' else sys.stdout
        output.write(colorize('base state: ', color='cyan', bold=True))
        output.write(str(self.state))
        output.write('\n')
        output.write(colorize('observation: ', color='blue'))
        output.write(str(self.obs))
        output.write('\n')
        if mode == 'ansi':
            return output

    def state2obs(self, s_id):
        s = np.zeros(self.dim, dtype='float32')
        s[:self.clutter_dim] = np.random.uniform(-self.confusion_level, self.confusion_level, size=self.clutter_dim)
        s[self.clutter_dim + s_id] = 1.
        s = np.dot(self.confusion, s)
        s = np.delete(s, self.clutter_dim + self.unobservable_states)  # remove unobservable states
        return s

    def write_mdp_to_dot(self, file_path='mdp.dot', overwrite=False):
        # To save DOT files as image files use for example: $ dot -T png -O mdp.dot
        if not os.path.isfile(file_path) or overwrite:
            with open(file_path, 'w') as writer:
                writer.write('digraph MDP {\n')
                for tr in self.transition_table:
                    writer.write(str(tr[0]) + ' -> ' + str(tr[2]) +
                                 ' [label="a:' + str(tr[1]) + ' ; p:' + str(tr[3]) + '"];\n')
                writer.write(str(self.init_state) + " [shape=diamond,color=lightblue,style=filled]\n")
                for node in self.good_terminals:
                    writer.write(str(node) + " [shape=box,color=green,style=filled]\n")
                for node in self.bad_terminals:
                    writer.write(str(node) + " [shape=box,color=red,style=filled]\n")
                writer.write('}')
        else:
            logger.warning('File "{0}" exists. Call with `overwrite=True` to permit overwrite.'.format(file_path))
