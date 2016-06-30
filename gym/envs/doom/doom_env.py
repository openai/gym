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
NUM_LEVELS = 9
CONFIG = 0
SCENARIO = 1
MAP = 2
DIFFICULTY = 3
ACTIONS = 4
MIN_SCORE = 5
TARGET_SCORE = 6

# Format (config, scenario, map, difficulty, actions, min, target)
DOOM_SETTINGS = [
    ['basic.cfg', 'basic.wad', 'map01', 5, [0, 10, 11], -485, 10],                                  # 0 - Basic
    ['deadly_corridor.cfg', 'deadly_corridor.wad', '', 1, [0, 10, 11, 13, 14, 15], -120, 1000],     # 1 - Corridor
    ['defend_the_center.cfg', 'defend_the_center.wad', '', 5, [0, 14, 15], -1, 10],                 # 2 - DefendCenter
    ['defend_the_line.cfg', 'defend_the_line.wad', '', 5, [0, 14, 15], -1, 15],                     # 3 - DefendLine
    ['health_gathering.cfg', 'health_gathering.wad', 'map01', 5, [13, 14, 15], 0, 1000],            # 4 - HealthGathering
    ['my_way_home.cfg', 'my_way_home.wad', '', 5, [13, 14, 15], -0.22, 0.5],                        # 5 - MyWayHome
    ['predict_position.cfg', 'predict_position.wad', 'map01', 3, [0, 14, 15], -0.075, 0.5],         # 6 - PredictPosition
    ['take_cover.cfg', 'take_cover.wad', 'map01', 5, [10, 11], 0, 750],                             # 7 - TakeCover
    ['deathmatch.cfg', 'deathmatch.wad', '', 5, list(range(NUM_ACTIONS)), 0, 20]                    # 8 - Deathmatch
]

class DoomEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 35}

    def __init__(self, level):
        utils.EzPickle.__init__(self)
        self.previous_level = -1
        self.level = level
        self.game = DoomGame()
        self.loader = Loader()
        self.doom_dir = os.path.dirname(os.path.abspath(__file__))
        self.mode = 'fast'                          # 'human', 'fast' or 'normal'
        self.no_render = False                      # To disable double rendering in human mode
        self.viewer = None
        self.is_initialized = False                 # Indicates that reset() has been called
        self.find_new_level = False                 # Indicates that we need a level change
        self.curr_seed  = 0
        self.action_space = spaces.HighLow(
            np.matrix([[0, 1, 0]] * 38 + [[-10, 10, 0]] * 2 + [[-100, 100, 0]] * 3, dtype=np.int8))
        self.allowed_actions = list(range(NUM_ACTIONS))
        self._configure()

    def _configure(self, screen_resolution=ScreenResolution.RES_640X480):
        # Often agents end up downsampling the observations. Configuring Doom to
        # return a smaller image yields significant (~10x) speedups
        if screen_resolution == ScreenResolution.RES_640X480:
            self.screen_height = 480
            self.screen_width = 640
            self.screen_resolution = ScreenResolution.RES_640X480
        elif screen_resolution == ScreenResolution.RES_160X120:
            self.screen_height = 120
            self.screen_width = 160
            self.screen_resolution = ScreenResolution.RES_160X120

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))

    def _load_level(self):
        # Closing if is_initialized
        if self.is_initialized:
            self.is_initialized = False
            self.game.close()
            self.game = DoomGame()

        # Loading Paths
        if not self.is_initialized:
            self.game.set_vizdoom_path(self.loader.get_vizdoom_path())
            self.game.set_doom_game_path(self.loader.get_freedoom_path())

        # Common settings
        self._closed = False
        self.game.load_config(os.path.join(self.doom_dir, 'assets/%s' % DOOM_SETTINGS[self.level][CONFIG]))
        self.game.set_doom_scenario_path(self.loader.get_scenario_path(DOOM_SETTINGS[self.level][SCENARIO]))
        if DOOM_SETTINGS[self.level][MAP] != '':
            self.game.set_doom_map(DOOM_SETTINGS[self.level][MAP])
        self.game.set_doom_skill(DOOM_SETTINGS[self.level][DIFFICULTY])
        self.previous_level = self.level
        self.allowed_actions = DOOM_SETTINGS[self.level][ACTIONS]
        self.game.set_screen_resolution(self.screen_resolution)

        # Algo mode
        if 'human' != self.mode:
            self.game.set_window_visible(False)
            self.game.set_mode(Mode.PLAYER)
            self.no_render = False
            self.game.init()
            self._start_episode()
            self.is_initialized = True
            return self.game.get_state().image_buffer.copy()

        # Human mode
        else:
            self.game.add_game_args('+freelook 1')
            self.game.set_window_visible(True)
            self.game.set_mode(Mode.SPECTATOR)
            self.no_render = True
            self.game.init()
            self._start_episode()
            self.is_initialized = True
            self._play_human_mode()
            return np.zeros(shape=self.observation_space.shape, dtype=np.uint8)

    def _start_episode(self):
        if self.curr_seed > 0:
            self.game.set_seed(self.curr_seed)
            self.curr_seed = 0
        self.game.new_episode()
        return

    def _play_human_mode(self):
        while not self.game.is_episode_finished():
            self.game.advance_action()
            state = self.game.get_state()
            total_reward = self.game.get_total_reward()
            info = self._get_game_variables(state.game_variables)
            info["TOTAL_REWARD"] = round(total_reward, 4)
            print('===============================')
            print('State: #' + str(state.number))
            print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
            print('Reward: \t' + str(self.game.get_last_reward()))
            print('Total Reward: \t' + str(total_reward))
            print('Variables: \n' + str(info))
            sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
        print('===============================')
        print('Done')
        return

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
            info = self._get_game_variables(state.game_variables)
            info["TOTAL_REWARD"] = round(self.game.get_total_reward(), 4)

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
            self._start_episode()
            return self.game.get_state().image_buffer.copy()
        else:
            return self._load_level()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None      # If we don't None out this reference pyglet becomes unhappy
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

    def _get_game_variables(self, state_variables):
        info = {}
        info["LEVEL"] = self.level
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
        return info

class MetaDoomEnv(DoomEnv):

    def __init__(self, average_over=10, passing_grade=600, min_tries_for_avg=5):
        super(MetaDoomEnv, self).__init__(0)
        self.average_over = average_over
        self.passing_grade = passing_grade
        self.min_tries_for_avg = min_tries_for_avg              # Need to use at least this number of tries to calc avg
        self.scores = [[]] * NUM_LEVELS
        self.locked_levels = [True] * NUM_LEVELS                # Locking all levels but the first
        self.locked_levels[0] = False
        self.total_reward = 0
        self._unlock_levels()

    def _play_human_mode(self):
        while not self.game.is_episode_finished():
            self.game.advance_action()
            state = self.game.get_state()
            episode_reward = self.game.get_total_reward()
            (reward, self.total_reward) = self._calculate_reward(episode_reward, self.total_reward)
            info = self._get_game_variables(state.game_variables)
            info["SCORES"] = self.get_scores()
            info["TOTAL_REWARD"] = round(self.total_reward, 4)
            info["LOCKED_LEVELS"] = self.locked_levels
            print('===============================')
            print('State: #' + str(state.number))
            print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
            print('Reward: \t' + str(reward))
            print('Total Reward: \t' + str(self.total_reward))
            print('Variables: \n' + str(info))
            sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
        print('===============================')
        print('Done')
        return

    def _get_next_level(self):
        # Finds the unlocked level with the lowest average
        averages = self.get_scores()
        lowest_level = 0                          # Defaulting to first level
        lowest_score = 1001
        for i in range(NUM_LEVELS):
            if not self.locked_levels[i]:
                if averages[i] < lowest_score:
                    lowest_level = i
                    lowest_score = averages[i]
        return lowest_level

    def _unlock_levels(self):
        averages = self.get_scores()
        for i in range(NUM_LEVELS - 2, -1, -1):
            if self.locked_levels[i + 1] and averages[i] >= self.passing_grade:
                self.locked_levels[i + 1] = False
        return

    def _start_episode(self):
        if 0 == len(self.scores[self.level]):
            self.scores[self.level] = [0] * self.min_tries_for_avg
        else:
            self.scores[self.level].insert(0, 0)
            self.scores[self.level] = self.scores[self.level][:self.min_tries_for_avg]
        return super(MetaDoomEnv, self)._start_episode()

    def change_level(self, new_level=None):
        if new_level is not None and self.locked_levels[new_level] == False:
            self.find_new_level = False
            self.level = new_level
            self.reset()
        else:
            self.find_new_level = False
            self.level = self._get_next_level()
            self.reset()
        return

    def _get_standard_reward(self, episode_reward):
        # Returns a standardized reward for an episode (i.e. between 0 and 1,000)
        min_score = DOOM_SETTINGS[self.level][MIN_SCORE]
        target_score = DOOM_SETTINGS[self.level][TARGET_SCORE]
        max_score = min_score + (target_score - min_score) / 0.99           # Target is 99th percentile (Scale 0-1000)
        std_reward = round(1000 * (episode_reward - min_score) / (max_score - min_score), 4)
        std_reward = min(1000, std_reward)                                  # Cannot be more than 1,000
        std_reward = max(0, std_reward)                                     # Cannot be less than 0
        return std_reward

    def get_total_reward(self):
        # Returns the sum of the average of all levels
        total_score = 0
        passed_levels = 0
        for i in range(NUM_LEVELS):
            if len(self.scores[i]) > 0:
                level_total = 0
                level_count = min(len(self.scores[i]), self.average_over)
                for j in range(level_count):
                    level_total += self.scores[i][j]
                level_average = level_total / level_count
                if level_average >= 990:
                    passed_levels += 1
                total_score += level_average
        # Bonus for passing all levels (50 * num of levels)
        if NUM_LEVELS == passed_levels:
            total_score += NUM_LEVELS * 50
        return round(total_score, 4)

    def _calculate_reward(self, episode_reward, prev_total_reward):
        # Calculates the action reward and the new total reward
        std_reward = self._get_standard_reward(episode_reward)
        self.scores[self.level][0] = std_reward
        total_reward = self.get_total_reward()
        reward = total_reward - prev_total_reward
        return reward, total_reward

    def get_scores(self):
        # Returns a list with the averages per level
        averages = [0] * NUM_LEVELS
        for i in range(NUM_LEVELS):
            if len(self.scores[i]) > 0:
                level_total = 0
                level_count = min(len(self.scores[i]), self.average_over)
                for j in range(level_count):
                    level_total += self.scores[i][j]
                level_average = level_total / level_count
                averages[i] = round(level_average, 4)
        return averages

    def _reset(self):
        # Reset is called on first step() after level is finished
        # or when change_level() is called. Returning if neither have been called to
        # avoid resetting the level twice
        if self.find_new_level:
            return

        if self.is_initialized and not self._closed and self.previous_level == self.level:
            self._start_episode()
            return self.game.get_state().image_buffer.copy()
        else:
            return self._load_level()

    def _step(self, action):
        # Changing level
        if self.find_new_level:
            self.change_level()

        if 'human' == self.mode:
            self._play_human_mode()
            obs = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
            reward = 0
            is_finished = True
            info = self._get_game_variables(None)
        else:
            obs, step_reward, is_finished, info = super(MetaDoomEnv, self)._step(action)
            reward, self.total_reward = self._calculate_reward(self.game.get_total_reward(), self.total_reward)

        info["SCORES"] = self.get_scores()
        info["TOTAL_REWARD"] = round(self.total_reward, 4)
        info["LOCKED_LEVELS"] = self.locked_levels

        # Indicating new level required
        if is_finished:
            self._unlock_levels()
            self.find_new_level = True

        return obs, reward, is_finished, info
