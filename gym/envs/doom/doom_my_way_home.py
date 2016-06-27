import logging
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)

class DoomMyWayHomeEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 6 - My Way Home ------------
    This map is designed to improve navigational skills. It is a series of
    interconnected rooms and 1 corridor with a dead end. Each room
    has a separate color. There is a green vest in one of the room.
    The vest is always in the same room. Player must find the vest.

    Allowed actions:
        [13] - MOVE_FORWARD                     - Move forward - Values 0 or 1
        [14] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [15] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Finding the vest
        -0.0001 - 35 times per second - Find the vest quick!

    Goal: 0.50 point
        Find the vest

    Mode:
        - env.mode can be 'fast', 'normal' or 'human' (e.g. env.mode = 'fast')
        - 'fast' (default) will run as fast as possible (~75 fps) (best for simulation)
        - 'normal' will run at roughly 35 fps (easier for human to watch)
        - 'human' will let you play the game (keyboard only: Arrow Keys, '<', '>' and Ctrl)

    Ends when:
        - Vest is found
        - Timeout (1 minutes - 2,100 frames)

    Actions:
        actions = [0] * 43
        actions[13] = 0      # MOVE_FORWARD
        actions[14] = 1      # TURN_RIGHT
        actions[15] = 0      # TURN_LEFT
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomMyWayHomeEnv, self).__init__(5)
