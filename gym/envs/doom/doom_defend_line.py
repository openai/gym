import logging
import os

import numpy as np

from doom_py import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader
from gym import error, spaces
from gym.envs.doom import doom_env
from gym.utils import seeding

logger = logging.getLogger(__name__)

class DoomDefendLineEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 4 - Defend the Line ------------
    This map is designed to teach you how to kill and how to stay alive.
    Your ammo will automatically replenish. You are only rewarded for kills,
    so figure out how to stay alive.

    The map is a rectangle with monsters in the middle. Monsters will
    respawn with additional health when killed. Kill as many as you can
    before they kill you. This map is harder than the previous.

    Allowed actions:
        [0]  - ATTACK                           - Shoot weapon - Values 0 or 1
        [13] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [14] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Killing the monster
        -  1    - Penalty for being killed

    Goal: 25 points
        Kill 25 monsters

    Ends when:
        - Player is dead
        - Timeout (60 seconds - 2100 frames)
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomDefendLineEnv, self).__init__()
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.loader = Loader()
        self.game = DoomGame()
        self.game.load_config(os.path.join(package_directory, 'assets/defend_the_line.cfg'))
        self.game.set_vizdoom_path(self.loader.get_vizdoom_path())
        self.game.set_doom_game_path(self.loader.get_freedoom_path())
        self.game.set_doom_scenario_path(self.loader.get_scenario_path('defend_the_line.wad'))
        self.screen_height = 480                    # Must match .cfg file
        self.screen_width = 640                     # Must match .cfg file
        self.game.set_window_visible(False)
        self.viewer = None
        # 3 allowed actions [0, 13, 14] (must match .cfg file)
        self.action_space = spaces.HighLow(np.matrix([[0, 1, 0]] * 3))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self._seed()
        self.game.init()
        self.game.new_episode()

    def _seed(self, seed=None):
        seed = seeding.hash_seed(seed) % 2**32
        self.game.set_seed(seed)
        return [seed]
