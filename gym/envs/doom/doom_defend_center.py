import logging
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)

class DoomDefendCenterEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 3 - Defend the Center ------------
    This map is designed to teach you how to kill and how to stay alive.
    You will also need to keep an eye on your ammunition level. You are only
    rewarded for kills, so figure out how to stay alive.

    The map is a circle with monsters. You are in the middle. Monsters will
    respawn with additional health when killed. Kill as many as you can
    before you run out of ammo.

    Allowed actions:
        [0]  - ATTACK                           - Shoot weapon - Values 0 or 1
        [14] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [15] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Killing a monster
        -  1    - Penalty for being killed

    Goal: 10 points
        Kill 11 monsters (you have 26 ammo)

    Mode:
        - env.mode can be 'fast', 'normal' or 'human' (e.g. env.mode = 'fast')
        - 'fast' (default) will run as fast as possible (~75 fps) (best for simulation)
        - 'normal' will run at roughly 35 fps (easier for human to watch)
        - 'human' will let you play the game (keyboard only: Arrow Keys, '<', '>' and Ctrl)

    Ends when:
        - Player is dead
        - Timeout (60 seconds - 2100 frames)

    Actions:
        actions = [0] * 43
        actions[0] = 0       # ATTACK
        actions[14] = 1      # TURN_RIGHT
        actions[15] = 0      # TURN_LEFT
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomDefendCenterEnv, self).__init__(2)
