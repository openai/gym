import logging
import os
from gym import error, spaces
import numpy as np
from gym.envs.doom import doom_env

try:
    from doom_py import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install Doom dependencies with 'pip install gym[doom].)'".format(e))

logger = logging.getLogger(__name__)

class DoomDefendCenterEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 3 - Defend the Center ------------
    This map is designed to teach you how to kill and how to stay alive.
    You will also need to keep an eye on your ammunition level. You are only
    rewarded for kills, so figure out how to stay alive.

    The map is a circle with monsters in the middle. Monsters will
    respawn with additional health when killed. Kill as many as you can
    before you run out of ammo.

    Allowed actions:
        [0]  - ATTACK                           - Shoot weapon - Values 0 or 1
        [13] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [14] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Killing the monster
        -  1    - Penalty for being killed

    Goal: 10 points
        Kill 10 monsters (you have 26 ammo)

    Ends when:
        - Player is dead
        - Timeout (60 seconds - 2100 frames)
    -----------------------------------------------------
    """
    def __init__(self):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.loader = Loader()
        self.game = DoomGame()
        self.game.load_config(os.path.join(package_directory, 'assets/defend_the_center.cfg'))
        self.game.set_vizdoom_path(self.loader.get_vizdoom_path())
        self.game.set_doom_game_path(self.loader.get_freedoom_path())
        self.game.set_doom_scenario_path(self.loader.get_scenario_path('defend_the_center.wad'))
        self.screen_height = 480                    # Must match .cfg file
        self.screen_width = 640                     # Must match .cfg file
        self.action_space = spaces.HighLow(np.matrix([[0, 1, 0]] * 36 + [[0, 10, 0]] * 5))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.allowed_actions = [0, 13, 14]          # Must match order in .cfg file
        self.game.set_window_visible(False)
        self.viewer = None
        self.sleep_time = 0.02857                   # 35 fps = 0.02857 sleep between frames
        self.game.init()
        self.game.new_episode()
