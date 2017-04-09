"""
MAME interface, based on atari_env
"""

import logging
logger = logging.getLogger(__name__)

import gym
from gym import error
from gym.spaces import Box, MultiDiscrete
from gym import utils
from gym.utils import seeding


try:
    import mamele
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install MAME dependencies by running 'pip install mamele')".format(e))

class MAMEEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}


    def __init__(self, game='galaxian', frameskip=(2, 5), watch=False):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int.

        If watch is True, we leave the video and sound on, and throttle to run realtime"""

        utils.EzPickle.__init__(self, game=game, frameskip=frameskip)

        self.frameskip = frameskip
        self.viewer = None
        self.previous_score = self.score = 0
        self.watch = watch

        self.mame = mamele.Mamele(game_name=game, watch=watch)

        self._seed()

        self.action_set = self.mame.get_minimal_action_set()
        self.initialise_action_space(self.action_set)

        screen_width, screen_height = self.mame.get_screen_dimensions()
        self._initialise_screen(screen_width, screen_height)


    def _initialise_screen(self, width, height):
        self.width = width
        self.height = height
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3))



    def initialise_action_space(self, action_spaces):

        # The description is a set of spaces, with the possible states for each of the spaces
        # eg [('vertical', ['noop', 'up', 'bottom']), ('coin1', ['noop', 'coin1']), ...]

        # We are meant to send back the state of each space as an action
        self.action_space = MultiDiscrete([[0, len(action_space[1])-1] for action_space in action_spaces])


    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]


    def _step(self, action):
        
        mapped_action = [space[1][index] for index, space in zip(action, self.action_set)]

        if isinstance(self.frameskip, int):
            steps = self.frameskip
        else:
            steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])


        reward = 0.0
        for _ in range(steps):
            reward += self.mame.act(mapped_action)

        return self._get_obs(), reward, self.mame.is_game_over(), {}
        

    @property
    def _n_actions(self):
        total = 1
        for subspace in self.action_set:
            total *= len(subspace[1])
        return total

    def _get_obs(self):    
        return self.mame.get_screen_rgb()

    # return: (states, observations)
    def _reset(self):
        self.mame.restart_game()
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        return self._get_obs()



    # def save_state(self):

    # def load_state(self):

    # def clone_state(self):

    # def restore_state(self, state):

