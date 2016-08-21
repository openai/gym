import logging
import os
import multiprocessing
import signal
import subprocess
import tempfile
from distutils import spawn
from threading import Thread, Lock
from time import sleep

import numpy as np

import gym
from gym import utils, spaces
from gym.utils import seeding

SEARCH_PATH = os.pathsep.join([os.environ['PATH'], '/usr/games', '/usr/local/games'])
FCEUX_PATH = spawn.find_executable('fceux', SEARCH_PATH)
if FCEUX_PATH is None:
    raise gym.error.DependencyNotInstalled("fceux is required. Try installing with apt-get install fceux.")

logger = logging.getLogger(__name__)

# Constants
NUM_ACTIONS = 6

# Singleton pattern
class NesLock:
    class __NesLock:
        def __init__(self):
            self.lock = multiprocessing.Lock()
    instance = None
    def __init__(self):
        if not NesLock.instance:
            NesLock.instance = NesLock.__NesLock()
    def get_lock(self):
        return NesLock.instance.lock


class NesEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.rom_path = ''
        self.screen_height = 224
        self.screen_width = 256
        self.action_space = spaces.HighLow(np.matrix([[0, 1, 0]] * NUM_ACTIONS, dtype=np.uint8))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.launch_vars = {}
        self.cmd_args = ['--xscale 2', '--yscale 2', '-f 0']
        self.lua_path = []
        self.subprocess = None
        self.no_render = True
        self.viewer = None

        # Pipes
        self.pipe_name = ''
        self.path_pipe_in = ''      # Input pipe (maps to fceux out-pipe and to 'in' file)
        self.path_pipe_out = ''     # Output pipe (maps to fceux in-pipe and to 'out' file)
        self.pipe_out = None
        self.lock_out = Lock()
        self.disable_in_pipe = False
        self.disable_out_pipe = False
        self.launch_vars['pipe_name'] = ''

        # Other vars
        self.is_initialized = 0     # Used to indicate fceux has been launched and is running
        self.is_exiting = 0         # Used to stop the listening thread
        self.last_frame = 0         # Last processed frame
        self.reward = 0             # Reward for last action
        self.episode_reward = 0     # Total rewards for episode
        self.is_finished = False
        self.screen = np.zeros(shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.info = {}
        self.level = 0
        self._reset_info_vars()
        self.first_step = False
        self.lock = (NesLock()).get_lock()

        # Seeding
        self.curr_seed = 0
        self._seed()

    def _configure(self, rom_path=None, lock=None):
        if rom_path is not None:
            self.rom_path = rom_path
        if lock is not None:
            self.lock = lock

    def _create_pipes(self):
        # Creates named pipe for inter-process communication
        self.pipe_name = seeding.hash_seed(None) % 2 ** 32
        self.launch_vars['pipe_name'] = self.pipe_name
        if not self.disable_out_pipe:
            self.path_pipe_out = os.path.join(tempfile.gettempdir(), 'smb-fifo-out.%d' % self.pipe_name)
            os.mkfifo(self.path_pipe_out)

        # Launching a thread that will listen to incoming pipe
        # Thread exits if self.is_exiting = 1 or pipe_in is closed
        if not self.disable_in_pipe:
            thread_incoming = Thread(target=self._listen_to_incoming_pipe, kwargs={'pipe_name': self.pipe_name})
            thread_incoming.start()

        # Cannot open output pipe now, otherwise it will block until
        # a reader tries to open the file in read mode - Must launch fceux first

    def _write_to_pipe(self, message):
        # Writes to output file (to communicate action to game)
        if self.disable_out_pipe or self.is_exiting == 1:
            return
        with self.lock_out:
            try:
                if self.pipe_out is None:
                    self.pipe_out = open(self.path_pipe_out, 'w', 1)
                self.pipe_out.write(message + '\n')
            except IOError:
                self.pipe_out = None

    def _close_pipes(self):
        # Closes named pipes
        with self.lock_out:
            if self.pipe_out is not None:
                pipe_out = self.pipe_out
                self.pipe_out = None
                try:
                    pipe_out.close()
                except BrokenPipeError:
                    pass
        if os.path.exists(self.path_pipe_out):
            try:
                os.remove(self.path_pipe_out)
            except OSError:
                pass
        self.pipe_name = ''
        self.launch_vars['pipe_name'] = self.pipe_name
        self.path_pipe_in = ''
        self.path_pipe_out = ''

    def _process_pipe_message(self, message):
        # To be overridden by game - Processes incoming messages
        pass

    def _listen_to_incoming_pipe(self, pipe_name):
        # Listens to incoming messages
        self.path_pipe_in = os.path.join(tempfile.gettempdir(), 'smb-fifo-in.%d' % pipe_name)
        if not os.path.exists(self.path_pipe_in):
            os.mkfifo(self.path_pipe_in)
        try:
            pipe_in = open(self.path_pipe_in, 'r', 1)
        except IOError:
            pipe_in = None
        buffer = ''
        while pipe_in is not None and 0 == self.is_exiting:
            # Readline sometimes break a line in 2
            # Using ! to indicate end of message
            message = pipe_in.readline().rstrip()
            if len(message) > 0:
                buffer += message
                if message[-1:-2:-1] == '!':
                    self._process_pipe_message(buffer[:-1])
                    if 'exit' == buffer[-5:-1]:
                        break
                    buffer = ''
        # Closing pipe
        if pipe_in is not None:
            try:
                pipe_in.close()
            except BrokenPipeError:
                pass
        if os.path.exists(self.path_pipe_in):
            try:
                os.remove(self.path_pipe_in)
            except OSError:
                pass
        self.is_exiting = 0

    def _launch_fceux(self):
        # Making sure ROM file is valid
        if '' == self.rom_path or not os.path.isfile(self.rom_path):
            raise gym.error.Error('Unable to find ROM. Please download the game from the web and configure the rom path by ' +
                                  'calling env.configure(rom_path=path_to_file)')

        # Creating pipes
        self._create_pipes()

        # Creating temporary lua file
        temp_lua_path = os.path.join('/tmp', str(seeding.hash_seed(None) % 2 ** 32) + '.lua')
        temp_lua_file = open(temp_lua_path, 'w', 1)
        for k, v in list(self.launch_vars.items()):
            temp_lua_file.write('%s = "%s";\n' % (k, v))
        i = 0
        for script in self.lua_path:
            temp_lua_file.write('f_%d = assert (loadfile ("%s"));\n' % (i, script))
            temp_lua_file.write('f_%d ();\n' % i)
            i += 1
        temp_lua_file.close()

        # Resetting variables
        self.last_frame = 0
        self.reward = 0
        self.episode_reward = 0
        self.is_finished = False
        self.screen = np.zeros(shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self._reset_info_vars()

        # Loading fceux
        args = [FCEUX_PATH]
        args.extend(self.cmd_args[:])
        args.extend(['--loadlua', temp_lua_path])
        args.append(self.rom_path)
        args.extend(['>/dev/null', '2>/dev/null', '&'])
        self.subprocess = subprocess.Popen(' '.join(args), shell=True)
        self.subprocess.communicate()
        if 0 == self.subprocess.returncode:
            self.is_initialized = 1
            if not self.disable_out_pipe:
                with self.lock_out:
                    try:
                        self.pipe_out = open(self.path_pipe_out, 'w', 1)
                    except IOError:
                        self.pipe_out = None
            # Removing lua file
            if os.path.isfile(temp_lua_path):
                try:
                    os.remove(temp_lua_path)
                except OSError:
                    pass
        else:
            self.is_initialized = 0
            raise gym.error.Error('Unable to start fceux. Command: %s' % (' '.join(args)))

    def _reset_info_vars(self):
        # Overridable - To reset the information variables
        self.info = {}

    def _start_episode(self):
        # Overridable - Starts a new episode
        return

    def _get_reward(self):
        # Overridable - Returns the reward for the last action
        return self.reward

    def _get_episode_reward(self):
        # Overridable - Returns the total reward earned for the episode
        return self.episode_reward

    def _get_is_finished(self):
        # Overridable - Returns a flag to indicate if the episode is finished
        return self.is_finished

    def _get_state(self):
        # Overridable - Returns the state
        return self.screen.copy()

    def _get_info(self):
        # Overridable - Returns the other variables
        return self.info

    def _step(self, action):
        if 0 == self.is_initialized:
            return self._get_state(), 0, self._get_is_finished(), {}

        if NUM_ACTIONS != len(action):
            logger.warn('NES action list must contain %d items. Padding missing items with 0' % NUM_ACTIONS)
            old_action = action
            action = [0] * NUM_ACTIONS
            for i in range(len(old_action)):
                action[i] = old_action[i]

        # Blocking until game sends ready
        loop_counter = 0
        restart_counter = 0
        if not self.disable_in_pipe:
            while 0 == self.last_frame:
                loop_counter += 1
                sleep(0.001)
                if 0 == self.is_initialized:
                    break
                if loop_counter >= 20000:
                    # Game not properly launched, relaunching
                    restart_counter += 1
                    loop_counter = 0
                    if restart_counter > 5:
                        self.close()
                        return self._get_state(), 0, True, {}
                    else:
                        self.reset()
                        sleep(5)

                elif loop_counter % 2500 == 0 and loop_counter > 4900:
                    # Incoming pipe not opened properly, reopening
                    thread_incoming = Thread(target=self._listen_to_incoming_pipe, kwargs={'pipe_name': self.pipe_name})
                    thread_incoming.start()

        start_frame = self.last_frame

        # Sending no-ops if in first step
        if self.first_step:
            self.first_step = False
            self.curr_seed = seeding.hash_seed(self.curr_seed) % 256
            self._write_to_pipe('noop_%d#%d' % (start_frame, self.curr_seed))

        # Sending commands and resetting reward to 0
        self.reward = 0
        self._write_to_pipe('commands_%d#%s' % (start_frame, ','.join([str(i) for i in action])))

        # Waiting for frame to be processed (self.last_frame will be increased when done)
        loop_counter = 0
        if not self.disable_in_pipe:
            while self.last_frame <= start_frame and not self.is_finished:
                loop_counter += 1
                sleep(0.001)
                if 0 == self.is_initialized:
                    break
                if loop_counter >= 20000:
                    # Game stuck, returning
                    # Likely caused by fceux incoming pipe not working
                    logger.warn('Closing episode (appears to be stuck). See documentation for how to handle this issue.')
                    if self.subprocess is not None:
                        # Workaround, killing process with pid + 1 (shell = pid, shell + 1 = fceux)
                        try:
                            os.kill(self.subprocess.pid + 1, signal.SIGKILL)
                        except OSError:
                            pass
                        self.subprocess = None
                    return self._get_state(), 0, True, {'ignore': True}

        # Getting results
        reward = self._get_reward()
        state = self._get_state()
        is_finished = self._get_is_finished()
        info = self._get_info()
        return state, reward, is_finished, info

    def _reset(self):
        if 1 == self.is_initialized:
            self.close()
        self.last_frame = 0
        self.reward = 0
        self.episode_reward = 0
        self.is_finished = False
        self.first_step = True
        self._reset_info_vars()
        with self.lock:
            self._launch_fceux()
            self._closed = False
            self._start_episode()
        self.screen = np.zeros(shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        return self._get_state()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                # If we don't None out this reference pyglet becomes unhappy
                self.viewer = None
            return
        if mode == 'human' and self.no_render:
            return
        img = self.screen.copy()  # Always rendering screen (as opposed to state)
        if img is None:
            img = np.zeros(shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def _close(self):
        # Terminating thread
        self.is_exiting = 1
        self._write_to_pipe('exit')
        sleep(0.05)
        if self.subprocess is not None:
            # Workaround, killing process with pid + 1 (shell = pid, shell + 1 = fceux)
            try:
                os.kill(self.subprocess.pid + 1, signal.SIGKILL)
            except OSError:
                pass
            self.subprocess = None
        sleep(0.001)
        self._close_pipes()
        self.last_frame = 0
        self.reward = 0
        self.episode_reward = 0
        self.is_finished = False
        self.screen = np.zeros(shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self._reset_info_vars()
        self.is_initialized = 0

    def _seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed) % 256
        return [self.curr_seed]

    def _get_rgb_from_palette(self, palette):
        rgb = {
            '00': (116, 116, 116),
            '01': (36, 24, 140),
            '02': (0, 0, 168),
            '03': (68, 0, 156),
            '04': (140, 0, 116),
            '05': (168, 0, 16),
            '06': (164, 0, 0),
            '07': (124, 8, 0),
            '08': (64, 44, 0),
            '09': (0, 68, 0),
            '0A': (0, 80, 0),
            '0B': (0, 60, 20),
            '0C': (24, 60, 92),
            '0D': (0, 0, 0),
            '0E': (0, 0, 0),
            '0F': (0, 0, 0),
            '10': (188, 188, 188),
            '11': (0, 112, 236),
            '12': (32, 56, 236),
            '13': (128, 0, 240),
            '14': (188, 0, 188),
            '15': (228, 0, 88),
            '16': (216, 40, 0),
            '17': (200, 76, 12),
            '18': (136, 112, 0),
            '19': (0, 148, 0),
            '1A': (0, 168, 0),
            '1B': (0, 144, 56),
            '1C': (0, 128, 136),
            '1D': (0, 0, 0),
            '1E': (0, 0, 0),
            '1F': (0, 0, 0),
            '20': (252, 252, 252),
            '21': (60, 188, 252),
            '22': (92, 148, 252),
            '23': (204, 136, 252),
            '24': (244, 120, 252),
            '25': (252, 116, 180),
            '26': (252, 116, 96),
            '27': (252, 152, 56),
            '28': (240, 188, 60),
            '29': (128, 208, 16),
            '2A': (76, 220, 72),
            '2B': (88, 248, 152),
            '2C': (0, 232, 216),
            '2D': (120, 120, 120),
            '2E': (0, 0, 0),
            '2F': (0, 0, 0),
            '30': (252, 252, 252),
            '31': (168, 228, 252),
            '32': (196, 212, 252),
            '33': (212, 200, 252),
            '34': (252, 196, 252),
            '35': (252, 196, 216),
            '36': (252, 188, 176),
            '37': (252, 216, 168),
            '38': (252, 228, 160),
            '39': (224, 252, 160),
            '3A': (168, 240, 188),
            '3B': (176, 252, 204),
            '3C': (156, 252, 240),
            '3D': (196, 196, 196),
            '3E': (0, 0, 0),
            '3F': (0, 0, 0),
            '40': (87, 87, 87),
            '41': (27, 18, 105),
            '42': (0, 0, 126),
            '43': (51, 0, 117),
            '44': (105, 0, 87),
            '45': (126, 0, 12),
            '46': (123, 0, 0),
            '47': (93, 6, 0),
            '48': (48, 33, 0),
            '49': (0, 51, 0),
            '4A': (0, 60, 0),
            '4B': (0, 45, 15),
            '4C': (18, 45, 69),
            '4D': (0, 0, 0),
            '4E': (0, 0, 0),
            '4F': (0, 0, 0),
            '50': (141, 141, 141),
            '51': (0, 84, 177),
            '52': (24, 42, 177),
            '53': (96, 0, 180),
            '54': (141, 0, 141),
            '55': (171, 0, 66),
            '56': (162, 30, 0),
            '57': (150, 57, 9),
            '58': (102, 84, 0),
            '59': (0, 111, 0),
            '5A': (0, 126, 0),
            '5B': (0, 108, 42),
            '5C': (0, 96, 102),
            '5D': (0, 0, 0),
            '5E': (0, 0, 0),
            '5F': (0, 0, 0),
            '60': (189, 189, 189),
            '61': (45, 141, 189),
            '62': (69, 111, 189),
            '63': (153, 102, 189),
            '64': (183, 90, 189),
            '65': (189, 87, 135),
            '66': (189, 87, 72),
            '67': (189, 114, 42),
            '68': (180, 141, 45),
            '69': (96, 156, 12),
            '6A': (57, 165, 54),
            '6B': (66, 186, 114),
            '6C': (0, 174, 162),
            '6D': (90, 90, 90),
            '6E': (0, 0, 0),
            '6F': (0, 0, 0),
            '70': (189, 189, 189),
            '71': (126, 171, 189),
            '72': (147, 159, 189),
            '73': (159, 150, 189),
            '74': (189, 147, 189),
            '75': (189, 147, 162),
            '76': (189, 141, 132),
            '77': (189, 162, 126),
            '78': (189, 171, 120),
            '79': (168, 189, 120),
            '7A': (126, 180, 141),
            '7B': (132, 189, 153),
            '7C': (117, 189, 180),
            '7D': (147, 147, 147),
            '7E': (0, 0, 0),
            '7F': (0, 0, 0),
        }
        if palette.upper() in rgb:
            return rgb[palette.upper()]
        else:
            return 0, 0, 0


class MetaNesEnv(NesEnv):
    def __init__(self, average_over=10, passing_grade=600, min_tries_for_avg=5, num_levels=0):
        NesEnv.__init__(self)
        self.average_over = average_over
        self.passing_grade = passing_grade
        self.min_tries_for_avg = min_tries_for_avg  # Need to use at least this number of tries to calc avg
        self.num_levels = num_levels
        self.scores = [[]] * self.num_levels
        self.locked_levels = [True] * self.num_levels  # Locking all levels but the first
        self.locked_levels[0] = False
        self.total_reward = 0
        self.find_new_level = False
        self._unlock_levels()

    def _get_next_level(self):
        # Finds the unlocked level with the lowest average
        averages = self.get_scores()
        lowest_level = 0  # Defaulting to first level
        lowest_score = 1001
        for i in range(self.num_levels):
            if not self.locked_levels[i]:
                if averages[i] < lowest_score:
                    lowest_level = i
                    lowest_score = averages[i]
        return lowest_level

    def _unlock_levels(self):
        averages = self.get_scores()
        for i in range(self.num_levels - 2, -1, -1):
            if self.locked_levels[i + 1] and averages[i] >= self.passing_grade:
                self.locked_levels[i + 1] = False
        return

    def _start_episode(self):
        if 0 == len(self.scores[self.level]):
            self.scores[self.level] = [0] * self.min_tries_for_avg
        else:
            self.scores[self.level].insert(0, 0)
            self.scores[self.level] = self.scores[self.level][:self.min_tries_for_avg]
        self.is_new_episode = True
        return NesEnv._start_episode(self)

    def change_level(self, new_level=None):
        self.find_new_level = False
        if new_level is not None and self.locked_levels[new_level] == False:
            self.level = new_level
        else:
            self.level = self._get_next_level()
        self._write_to_pipe('changelevel#' + str(self.level))
        self.reset()

    def _get_standard_reward(self, episode_reward):
        # Can be overridden
        std_reward = episode_reward
        std_reward = min(1000, std_reward)                                  # Cannot be more than 1,000
        std_reward = max(0, std_reward)                                     # Cannot be less than 0
        return std_reward

    def get_total_reward(self):
        # Returns the sum of the average of all levels
        total_score = 0
        passed_levels = 0
        for i in range(self.num_levels):
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
        if self.num_levels == passed_levels:
            total_score += self.num_levels * 50
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
        averages = [0] * self.num_levels
        for i in range(self.num_levels):
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

        self.last_frame = 0
        self.reward = 0
        self.episode_reward = 0
        self.is_finished = False
        self._reset_info_vars()
        if 0 == self.is_initialized:
            self._launch_fceux()
            self._closed = False
        self._start_episode()
        self.screen = np.zeros(shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        return self._get_state()

    def _step(self, action):
        # Changing level
        if self.find_new_level:
            self.change_level()

        obs, step_reward, is_finished, info = NesEnv._step(self, action)
        reward, self.total_reward = self._calculate_reward(self._get_episode_reward(), self.total_reward)
        # First step() after new episode returns the entire total reward
        # because stats_recorder resets the episode score to 0 after reset() is called
        if self.is_new_episode:
            reward = self.total_reward

        self.is_new_episode = False
        info["level"] = self.level
        info["scores"] = self.get_scores()
        info["total_reward"] = round(self.total_reward, 4)
        info["locked_levels"] = self.locked_levels

        # Indicating new level required
        if is_finished:
            self._unlock_levels()
            self.find_new_level = True

        return obs, reward, is_finished, info
