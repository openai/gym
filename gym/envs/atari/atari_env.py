import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils

try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install Atari dependencies with 'pip install gym[atari].)'".format(e))

import logging
logger = logging.getLogger(__name__)

def to_rgb(ale):
    (screen_width,screen_height) = ale.getScreenDims()
    arr = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    ale.getScreenRGB(arr) # says rgb but actually bgr
    return arr[:,:,[2, 1, 0]].copy()

def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)
    return ram

class AtariEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game='pong', obs_type='ram'):
        utils.EzPickle.__init__(self, game, obs_type)
        assert obs_type in ('ram', 'image')
        game_path = atari_py.get_game_path(game)
        if not os.path.exists(game_path):
            raise IOError('You asked for game %s but path %s does not exist'%(game, game_path))
        self.ale = atari_py.ALEInterface()
        self.ale.loadROM(game_path)
        self._obs_type = obs_type
        self._action_set = self.ale.getMinimalActionSet()
        self.viewer = None

        (screen_width,screen_height) = self.ale.getScreenDims()

        self.action_space = spaces.Discrete(len(self._action_set))
        if self._obs_type == 'ram':
            self.observation_space = spaces.Box(low=np.zeros(128), high=np.zeros(128)+255)
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def _step(self, a):
        reward = 0.0
        action = self._action_set[a]
        num_steps = np.random.randint(2, 5)
        for _ in range(num_steps):
            reward += self.ale.act(action)
        ob = self._get_obs()

        return ob, reward, self.ale.game_over(), {}

    def _get_image(self):
        return to_rgb(self.ale)
    def _get_ram(self):
        return to_ram(self.ale)

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == 'ram':
            return self._get_ram()
        elif self._obs_type == 'image':
            img = self._get_image()
        return img

    # return: (states, observations)
    def _reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]



ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}
