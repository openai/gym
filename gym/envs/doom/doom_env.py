import logging, os
from time import sleep

import numpy as np

import gym
from gym import utils, spaces
from gym.utils import seeding

try:
    import doom_py
    from doom_py import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader
except ImportError as e:
    raise gym.error.DependencyNotInstalled("{}. (HINT: you can install Doom dependencies with 'pip install gym[doom].)'".format(e))

logger = logging.getLogger(__name__)

# Constants
NUM_ACTIONS = 43

class DoomEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 35}

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.game = DoomGame()
        self.mode = 'fast'                          # 'human', 'fast' or 'normal'
        self.no_render = False                      # To disable double rendering in human mode
        self.config = ''                            # Configuration file
        self.scenario = ''                          # Scenario file
        self.map = ''                               # Map
        self.difficulty = 5                         # Difficulty (1 = Easy, 10 = Impossible)
        self.viewer = None
        self.is_initialized = False                 # Indicates that reset() has been called
        self.curr_seed  = 0
        self.screen_height = 480
        self.screen_width = 640
        self.action_space = spaces.HighLow(
            np.matrix([[0, 1, 0]] * 38 + [[-10, 10, 0]] * 2 + [[-100, 100, 0]] * 3, dtype=np.int8))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.allowed_actions = list(range(NUM_ACTIONS))

    def _step(self, action):
        if NUM_ACTIONS != len(action):
            logger.warn('Doom action list must contain %d items. Padding missing items with 0' % NUM_ACTIONS)
            old_action = action
            action = [0] * NUM_ACTIONS
            for i in range(len(old_action)):
                action[i] = old_action[i]
        # action is a list of numbers but DoomGame.make_action expects a list of ints
        if len(self.allowed_actions) > 0:
            list_action = [int(action[action_idx]) for action_idx in self.allowed_actions]
        else:
            list_action = [int(x) for x in action]
        try:
            reward = self.game.make_action(list_action)
            state = self.game.get_state()
            info = self._get_game_variables(state.game_variables, self.game.get_total_reward())

            if self.game.is_episode_finished():
                is_finished = True
                return np.zeros(shape=self.observation_space.shape, dtype=np.uint8), reward, is_finished, info
            else:
                is_finished = False
                return state.image_buffer.copy(), reward, is_finished, info

        except doom_py.vizdoom.ViZDoomIsNotRunningException:
            return np.zeros(shape=self.observation_space.shape, dtype=np.uint8), 0, True, {}

    def _reset(self):
        if self.is_initialized and not self._closed:
            if self.curr_seed > 0:
                self.game.set_seed(self.curr_seed)
            self.game.new_episode()
            return self.game.get_state().image_buffer.copy()
        self._closed = False
        package_directory = os.path.dirname(os.path.abspath(__file__))

        # Common settings
        self.loader = Loader()
        if self.config != '':
            self.game.load_config(os.path.join(package_directory, 'assets/%s' % self.config))
        self.game.set_vizdoom_path(self.loader.get_vizdoom_path())
        self.game.set_doom_game_path(self.loader.get_freedoom_path())
        if self.scenario != '':
            self.game.set_doom_scenario_path(self.loader.get_scenario_path(self.scenario))
        if self.map != '':
            self.game.set_doom_map(self.map)
        self.game.set_doom_skill(self.difficulty)

        # Algo mode
        if 'human' != self.mode:
            self.game.set_window_visible(False)
            self.game.set_mode(Mode.PLAYER)
            self.no_render = False
            self.game.init()
            if self.curr_seed > 0:
                self.game.set_seed(self.curr_seed)
            self.game.new_episode()
            self.is_initialized = True
            return self.game.get_state().image_buffer.copy()

        # Human mode
        else:
            self.game.add_game_args('+freelook 1')
            self.game.set_window_visible(True)
            self.game.set_mode(Mode.SPECTATOR)
            self.no_render = True
            self.game.init()
            if self.curr_seed > 0:
                self.game.set_seed(self.curr_seed)
            self.game.new_episode()
            self.is_initialized = True

            while not self.game.is_episode_finished():
                self.game.advance_action()
                state = self.game.get_state()
                info = state.game_variables
                total_reward = self.game.get_total_reward()
                print('===============================')
                print('State: #' + str(state.number))
                print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
                print('Reward: \t' + str(self.game.get_last_reward()))
                print('Total Reward: \t' + str(total_reward))
                print('Variables: \n' + str(self._get_game_variables(info, total_reward)))
                sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
            print('===============================')
            print('Done')
            return

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                # If we don't None out this reference pyglet becomes unhappy
                self.viewer = None
            return
        try:
            if 'human' == mode and self.no_render: return
            state = self.game.get_state()
            img = state.image_buffer
            # VizDoom returns None if the episode is finished, let's make it
            # an empty image so the recorder doesn't stop
            if img is None:
                img = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
            if mode == 'rgb_array':
                return img
            elif mode is 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
                if 'normal' == self.mode:
                    sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
        except doom_py.vizdoom.ViZDoomIsNotRunningException:
            pass # Doom has been closed

    def _close(self):
        self.game.close()

    def _seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed) % 2 ** 32
        return [ self.curr_seed ]

    def _get_game_variables(self, state_variables, total_reward):
        info = {}
        if state_variables is None: return info
        info['KILLCOUNT'] = state_variables[0]
        info['ITEMCOUNT'] = state_variables[1]
        info['SECRETCOUNT'] = state_variables[2]
        info['FRAGCOUNT'] = state_variables[3]
        info['HEALTH'] = state_variables[4]
        info['ARMOR'] = state_variables[5]
        info['DEAD'] = state_variables[6]
        info['ON_GROUND'] = state_variables[7]
        info['ATTACK_READY'] = state_variables[8]
        info['ALTATTACK_READY'] = state_variables[9]
        info['SELECTED_WEAPON'] = state_variables[10]
        info['SELECTED_WEAPON_AMMO'] = state_variables[11]
        info['AMMO1'] = state_variables[12]
        info['AMMO2'] = state_variables[13]
        info['AMMO3'] = state_variables[14]
        info['AMMO4'] = state_variables[15]
        info['AMMO5'] = state_variables[16]
        info['AMMO6'] = state_variables[17]
        info['AMMO7'] = state_variables[18]
        info['AMMO8'] = state_variables[19]
        info['AMMO9'] = state_variables[20]
        info['AMMO0'] = state_variables[21]
        info['TOTAL_REWARD'] = total_reward
        return info