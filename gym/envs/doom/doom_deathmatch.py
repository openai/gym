import logging
import os

import numpy as np

from doom_py import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader
from gym import spaces
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)

class DoomDeathmatchEnv(doom_env.DoomEnv):
    """
    ------------ Final Mission - Deathmatch ------------
    Kill as many monsters as possible without being killed.

    Allowed actions:
        ALL
    Note: see controls.md for details

    Rewards:
        +1      - Killing a monster

    Goal: 20 points
        Kill 20 monsters without being killed

    Ends when:
        - Player is dead
        - Timeout (3 minutes - 6,300 frames)

    Actions:
       actions = [0] * 43
       actions[0] = 0       # ATTACK
       actions[1] = 0       # USE
       [...]
       actions[42] = 0      # MOVE_UP_DOWN_DELTA
       A full list of possible actions is available in controls.md
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomDeathmatchEnv, self).__init__()
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.loader = Loader()
        self.game = DoomGame()
        self.game.load_config(os.path.join(package_directory, 'assets/deathmatch.cfg'))
        self.game.set_vizdoom_path(self.loader.get_vizdoom_path())
        self.game.set_doom_game_path(self.loader.get_freedoom_path())
        self.game.set_doom_scenario_path(self.loader.get_scenario_path('deathmatch.wad'))
        self.screen_height = 480                    # Must match .cfg file
        self.screen_width = 640                     # Must match .cfg file
        self.game.set_window_visible(False)
        self.viewer = None
        self.game.init()
        self.game.new_episode()

        # 43 allowed actions (must match .cfg file)
        # [0 to 37 are either 0 or 1, 38 and 39 are -10 to +10, 40 to 42 are -100 to +100]
        self.action_space = spaces.HighLow(np.matrix([[0, 1, 0]] * 38 + [[-10, 10, 0]] * 2 + [[-100, 100, 0]] * 3))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.action_space.allowed_actions = list(range(43))

        self._seed()
