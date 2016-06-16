import logging
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)

class DoomDefendLineEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 4 - Defend the Line ------------
    This map is designed to teach you how to kill and how to stay alive.
    Your ammo will automatically replenish. You are only rewarded for kills,
    so figure out how to stay alive.

    The map is a rectangle with monsters on the other side. Monsters will
    respawn with additional health when killed. Kill as many as you can
    before they kill you. This map is harder than the previous.

    Allowed actions:
        [0]  - ATTACK                           - Shoot weapon - Values 0 or 1
        [14] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [15] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Killing a monster
        -  1    - Penalty for being killed

    Goal: 15 points
        Kill 16 monsters

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
        super(DoomDefendLineEnv, self).__init__(3)
