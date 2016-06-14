import logging
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)

class DoomTakeCoverEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 8 - Take Cover ------------
    This map is to train you on the damage of incoming missiles.
    It is a rectangular map with monsters firing missiles and fireballs
    at you. You need to survive as long as possible.

    Allowed actions:
        [10] - MOVE_RIGHT                       - Move to the right - Values 0 or 1
        [11] - MOVE_LEFT                        - Move to the left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - 35 times per second - Survive as long as possible

    Goal: 750 points
        Survive for ~ 20 seconds

    Mode:
        - env.mode can be 'fast', 'normal' or 'human' (e.g. env.mode = 'fast')
        - 'fast' (default) will run as fast as possible (~75 fps) (best for simulation)
        - 'normal' will run at roughly 35 fps (easier for human to watch)
        - 'human' will let you play the game (keyboard only: Arrow Keys, '<', '>' and Ctrl)

    Ends when:
        - Player is dead (one or two fireballs should be enough to kill you)
        - Timeout (60 seconds - 2,100 frames)

    Actions:
        actions = [0] * 43
        actions[10] = 0      # MOVE_RIGHT
        actions[11] = 1      # MOVE_LEFT
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomTakeCoverEnv, self).__init__(7)
