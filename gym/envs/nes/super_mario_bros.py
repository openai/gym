import logging
import os

import numpy as np

from gym import spaces
from gym.envs.nes import nes_env

logger = logging.getLogger(__name__)

# (world_number, level_number, area_number, max_distance)
WORLD_NUMBER = 0
LEVEL_NUMBER = 1
AREA_NUMBER = 2
MAX_DISTANCE = 3
SMB_LEVELS = [
    (1, 1, 1, 3266), (1, 2, 2, 3266), (1, 3, 4, 2514), (1, 4, 5, 2430),
    (2, 1, 1, 3298), (2, 2, 2, 3266), (2, 3, 4, 3682), (2, 4, 5, 2430),
    (3, 1, 1, 3298), (3, 2, 2, 3442), (3, 3, 3, 2498), (3, 4, 4, 2430),
    (4, 1, 1, 3698), (4, 2, 2, 3266), (4, 3, 4, 2434), (4, 4, 5, 2942),
    (5, 1, 1, 3282), (5, 2, 2, 3298), (5, 3, 3, 2514), (5, 4, 4, 2429),
    (6, 1, 1, 3106), (6, 2, 2, 3554), (6, 3, 3, 2754), (6, 4, 4, 2429),
    (7, 1, 1, 2962), (7, 2, 2, 3266), (7, 3, 4, 3682), (7, 4, 5, 3453),
    (8, 1, 1, 6114), (8, 2, 2, 3554), (8, 3, 3, 3554), (8, 4, 4, 4989)]
SUPER_MARIO_ROM_PATH = os.path.expanduser('~/.roms') + '/super-mario.nes'

class SuperMarioBrosEnv(nes_env.NesEnv):
    """
    ------------ NES - Super Mario Bros ------------

    =====================
      Single Level
    =====================

    This environment allows you to play the original Super Mario Bros.

    Environments:
        - There are 64 environments available, with the following syntax:

            SuperMarioBros-<world_number>-<level_number>-v0
             and
            SuperMarioBros-<world_number>-<level_number>-Tiles-v0

            - world_number is a number between 1 and 8
            - level_number is a number between 1 and 4

            e.g. SuperMarioBros-6-1-v0, or SuperMarioBros-3-4-Tiles-v0

    Tiles vs Regular:
        - Environment with "Tiles" in their name will return a 16x13 array
          representation of the screen, where each square can have one of
          the following values:

          - 0: empty space
          - 1: object (e.g. platform, coins, flagpole, etc.)
          - 2: enemy
          - 3: Mario

        - Environment without "Tiles" will return a 256x224 array representation
          of the screen, where each square contains red, blue, and green value (RGB)

        - "Tiles" environment can also access the screen information by calling

            screen = env.render(mode='rgb_array')

    Actions:
        - The NES controller is composed of 6 buttons (Up, Left, Down, Right, A, B)
        - The step function expects an array of 0 and 1 that represents

            - First Item -  Up
            - Second Item - Left
            - Third Item -  Down
            - Fourth Item - Right
            - Fifth Item -  A
            - Sixth Item -  B

        e.g. action = [0, 0, 0, 1, 1, 0]    # [up, left, down, right, A, B]
        would activate right (4th element), and A (5th element)

        - An action of '1' represents a key down, and '0' a key up.
        - To toggle the button, you must issue a key up, then a key down.

    Mode:
        - The environment can be initialized with 3 different modes:

        - "human" - Initializes fceux so a human can play the game (inputs from step() are ignored)
        - "normal" - Initializes fceux with default rendering speed
        - "fast" (default) - Initializes fceux to run the emulation as fast as possible

    ROM:
        - This environment requires the Super Mario Bros ROM to load
        - The ROM is not provided with OpenAI, but can easily be downloaded from the internet
        - A Google search for "fceux rom Super Mario Bros" should guide you in the right direction
        - The ROM path needs to be set with env.configure(rom_path='rom_path')
        - You can also put the rom in $HOME/.rom/super-mario.nes to automatically load it
        - The following script should install it automatically:

        mkdir -p ~/.roms && curl "https://storage.googleapis.com/ppaquette-files/openai/super-mario.nes" > ~/.roms/super-mario.nes

    Initiating the environment:
        - SuperMarioBros can be initiated with:

            import gym
            env = gym.make('SuperMarioBros-1-1-v0')
            env.configure(rom_path='/path/to/rom/Super Mario Bros. (Japan, USA).nes')
            env.reset()

        - fceux will be launched when reset() is called

    Gameplay:
        - The game will automatically close if Mario dies or shortly after the flagpole is touched
        - The game will only accept inputs after the timer has started to decrease (i.e. it will automatically move
          through the menus and animations)
        - The total reward is the distance on the x axis.

    Rendering:
        - render() will not generate a 2nd rendering, because fceux is already doing so
        - to disable this behaviour and have render() generate a separate rendering, set env.no_render = False

    Variables:
        - The following variables are available in the info dict

            - distance        # Total distance from the start (x-axis)
            - life            # Number of lives Mario has (should always be 3)
            - score           # The current score
            - coins           # The current number of coins
            - time            # The current time left
            - player_status   # Indicates if Mario is small (value of 0), big (value of 1), or can shoot fireballs (2+)

            - ignore          # Will be added with a value of True if the game is stuck and is terminated early

        - A value of -1 indicates that the value is unknown

    Configuration:
        After creating the env, you can call env.configure() to configure some parameters.

        - lock [e.g. env.configure(lock=multiprocessing_lock)]
            SuperMario requires a multiprocessing lock when running across multiple processes, otherwise the game might get stuck

            You can either:

            1) [Preferred] Create a multiprocessing.Lock() and pass it as a parameter to the configure() method
                [e.g. env.configure(lock=multiprocessing_lock)]

            2) Create and close an environment before running your multiprocessing routine, this will create
                a singleton lock that will be cached in memory, and be used by all SuperMario environments afterwards
                [e.g. env = gym.make('SuperMarioBros-...'); env.close()]

            3) Manually wrap calls to reset() and close() in a multiprocessing.Lock()

    Game is stuck:
        - In some cases, it is possible for the game to become stuck. This is likely due to a named pipe not working properly.

        - To reduce these issues, try to pass a lock to the configure method (see above), and try to reduce the number of
          running processes.

        - After 20 seconds, the stuck game will be automatically closed, and step() will return done=True with an info
          dictionary containing ignore=True. You can simply check if the ignore key is in the info dictionary, and ignore
          that specific episode.

    =====================
      META Level
    =====================

    Goal: 32,000 points
        - Pass all levels

    Scoring:
        - Each level score has been standardized on a scale of 0 to 1,000
        - The passing score for a level is 990 (99th percentile)
        - A bonus of 1,600 (50 * 32 levels) is given if all levels are passed
        - The score for a level is the average of the last 3 tries
        - If there has been less than 3 tries for a level, the missing tries will have a score of 0
          (e.g. if you score 1,000 on the first level on your first try, your level score will be
          (1,000 + 0 + 0) / 3 = 333.33)
        - The total score is the sum of the level scores, plus the bonus if you passed all levels.

        e.g. List of tries:

        - Level 0: 500
        - Level 0: 750
        - Level 0: 800
        - Level 0: 1,000
        - Level 1: 100
        - Level 1: 200

        Level score for level 0 = [1,000 + 800 + 750] / 3 = 850     (Average of last 3 tries)
        Level score for level 1 = [200 + 100 + 0] / 3 = 100         (Tries not completed have a score of 0)
        Level score for levels 2 to 8 = 0
        Bonus score for passing all levels = 0
        ------------------------
        Total score = 850 + 100 + 0 + 0 = 950

    Changing Level:
        - To unlock the next level, you must achieve a level score (avg of last 3 tries) of at least 600
          (i.e. passing 60% of the last level)
        - There are 2 ways to change level:

        1) Manual method

            - obs, reward, is_finished, info = env.step(action)
            - if is_finished is true, you can call env.change_level(level_number) to change to an unlocked level
            - level_number is a number from 0 to 31
            - you can see
                the current level with info["level"]
                the list of level score with info["scores"],
                the list of locked levels with info["locked_levels"]
                your total score with info["total_reward"]

            e.g.
                import gym
                env = gym.make('meta-SuperMarioBros-v0')
                env.reset()
                total_score = 0
                while total_score < 32000:
                    action = [0] * 6
                    obs, reward, is_finished, info = env.step(action)
                    env.render()
                    total_score = info["total_reward"]
                    if is_finished:
                        env.change_level(level_you_want)

        2) Automatic change

            - if you don't call change_level() and the level is finished, the system will automatically select the
              unlocked level with the lowest level score (which is likely to be the last unlocked level)

            e.g.
                import gym
                env = gym.make('meta-SuperMarioBros-v0')
                env.reset()
                total_score = 0
                while total_score < 32000:
                    action = [0] * 6
                    obs, reward, is_finished, info = env.step(action)
                    env.render()
                    total_score = info["total_reward"]

    -----------------------------------------------------
    """
    def __init__(self, draw_tiles=0, level=0):
        nes_env.NesEnv.__init__(self)
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.level = level
        self.draw_tiles = draw_tiles
        self._mode = 'fast'
        self.lua_path.append(os.path.join(package_directory, 'lua/super-mario-bros.lua'))
        self.tiles = None
        self.launch_vars['target'] = self._get_level_code(self.level)
        self.launch_vars['mode'] = 'fast'
        self.launch_vars['meta'] = '0'
        self.launch_vars['draw_tiles'] = str(draw_tiles)
        if os.path.isfile(SUPER_MARIO_ROM_PATH):
            self.rom_path = SUPER_MARIO_ROM_PATH

        # Tile mode
        if 1 == self.draw_tiles:
            self.tile_height = 13
            self.tile_width = 16
            self.tiles = np.zeros(shape=(self.tile_height, self.tile_width), dtype=np.uint8)
            self.observation_space = spaces.Box(low=0, high=3, shape=(self.tile_height, self.tile_width))

    # --------------
    # Properties
    # --------------
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        self.launch_vars['mode'] = value
        self.cmd_args = ['--xscale 2', '--yscale 2', '-f 0']
        if 'human' == value:
            self.disable_out_pipe = True
            self.disable_in_pipe = True
        else:
            self.disable_out_pipe = False
            self.disable_in_pipe = False

    # --------------
    # Methods
    # --------------
    def _get_level_code(self, level):
        world_number = int(level / 4) + 1
        level_number = (level % 4) + 1
        area_number = level_number
        # Worlds 1, 2, 4, 7 have a transition as area number 2 (so 2-2 is area 3 and 3, 2-3 is area 4, 2-4 is area 5)
        if world_number in [1, 2, 4, 7] and level_number >= 2:
            area_number += 1
        return '%d%d%d' % (world_number, level_number, area_number)

    def _process_data_message(self, frame_number, data):
        # Format: data_<frame>#name_1:value_1|name_2:value_2|...
        if frame_number <= self.last_frame or self.info is None:
            return
        parts = data.split('|')
        for part in parts:
            if part.find(':') == -1:
                continue
            parts_2 = part.split(':')
            name = parts_2[0]
            value = int(parts_2[1])
            if 'is_finished' == name:
                self.is_finished = bool(value)
            elif 'distance' == name:
                self.reward = value - self.info[name]
                self.episode_reward = value
                self.info[name] = value
            else:
                self.info[name] = value

    def _process_screen_message(self, frame_number, data):
        # Format: screen_<frame>#<x (2 hex)><y (2 hex)><palette (2 hex)>|<x><y><p>|...
        if frame_number <= self.last_frame or self.screen is None:
            return
        parts = data.split('|')
        for part in parts:
            if 6 == len(part):
                x = int(part[0:2], 16)
                y = int(part[2:4], 16)
                self.screen[y][x] = self._get_rgb_from_palette(part[4:6])

    def _process_tiles_message(self, frame_number, data):
        # Format: tiles_<frame>#<x (1 hex)><y (1 hex)><value (1 hex)>|<x><y><v>|...
        if frame_number <= self.last_frame or self.tiles is None:
            return
        parts = data.split('|')
        for part in parts:
            if 3 == len(part):
                x = int(part[0:1], 16)
                y = int(part[1:2], 16)
                v = int(part[2:3], 16)
                self.tiles[y][x] = v

    def _process_ready_message(self, frame_number):
        # Format: ready_<frame>
        if 0 == self.last_frame:
            self.last_frame = frame_number

    def _process_done_message(self, frame_number):
        # Done means frame is done processing, please send next command
        # Format: done_<frame>
        if frame_number > self.last_frame:
            self.last_frame = frame_number

    def _process_reset_message(self):
        # Reset means 'changelevel' needs to be sent and last_frame needs to be set to 0
        # Not implemented in non-meta levels
        pass

    def _process_exit_message(self):
        # Exit means fceux is terminating
        # Format: exit
        self.is_finished = True
        self._is_exiting = 1
        self.close()

    def _parse_frame_number(self, parts):
        # Parsing frame number
        try:
            frame_number = int(parts[1]) if len(parts) > 1 else 0
            return frame_number
        except:
            pass

        # Sometimes beginning of message is sent twice (screen_70screen_707#)
        if len(parts) > 2 and parts[2].isdigit():
            tentative_frame = int(parts[2])
            if self.last_frame - 10 < tentative_frame < self.last_frame + 10:
                return tentative_frame

        # Otherwise trying to make sense of digits
        else:
            digits = ''.join(c for c in ''.join(parts[1:]) if c.isdigit())
            tentative_frame = int(digits) if len(digits) > 1 else 0
            if self.last_frame - 10 < tentative_frame < self.last_frame + 10:
                return tentative_frame

        # Unable to parse - Likely an invalid message
        return None

    def _process_pipe_message(self, message):
        # Parsing
        parts = message.split('#')
        header = parts[0] if len(parts) > 0 else ''
        data = parts[1] if len(parts) > 1 else ''
        parts = header.split('_')
        message_type = parts[0] if len(parts) > 0 else ''
        frame_number = self._parse_frame_number(parts)

        # Invalid message - Ignoring
        if frame_number is None:
            return

        # Processing
        if 'data' == message_type:
            self._process_data_message(frame_number, data)
        elif 'screen' == message_type:
            self._process_screen_message(frame_number, data)
        elif 'tiles' == message_type:
            self._process_tiles_message(frame_number, data)
        elif 'ready' == message_type:
            self._process_ready_message(frame_number)
        elif 'done' == message_type:
            self._process_done_message(frame_number)
        elif 'reset' == message_type:
            self._process_reset_message()
        elif 'exit' == message_type:
            self._process_exit_message()

    def _get_reward(self):
        return self.reward

    def _get_episode_reward(self):
        return self.episode_reward

    def _get_is_finished(self):
        return self.is_finished

    def _get_state(self):
        if 1 == self.draw_tiles:
            return self.tiles.copy()
        else:
            return self.screen.copy()

    def _get_info(self):
        return self.info

    def _reset_info_vars(self):
        self.info = {
            'level': self.level,
            'distance': 0,
            'score': -1,
            'coins': -1,
            'time': -1,
            'player_status': -1
        }


class MetaSuperMarioBrosEnv(SuperMarioBrosEnv, nes_env.MetaNesEnv):

    def __init__(self, average_over=10, passing_grade=600, min_tries_for_avg=5, draw_tiles=0):
        nes_env.MetaNesEnv.__init__(self,
                                    average_over=average_over,
                                    passing_grade=passing_grade,
                                    min_tries_for_avg=min_tries_for_avg,
                                    num_levels=32)
        SuperMarioBrosEnv.__init__(self, draw_tiles=draw_tiles, level=0)
        self.launch_vars['meta'] = '1'

    def _process_reset_message(self):
        self.last_frame = 0

    def _get_standard_reward(self, episode_reward):
        # Returns a standardized reward for an episode (i.e. between 0 and 1,000)
        min_score = 0
        target_score = float(SMB_LEVELS[self.level][MAX_DISTANCE]) - 40
        max_score = min_score + (target_score - min_score) / 0.99  # Target is 99th percentile (Scale 0-1000)
        std_reward = round(1000 * (episode_reward - min_score) / (max_score - min_score), 4)
        std_reward = min(1000, std_reward)  # Cannot be more than 1,000
        std_reward = max(0, std_reward)  # Cannot be less than 0
        return std_reward
