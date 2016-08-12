import logging
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)


class DoomPredictPositionEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 7 - Predict Position ------------
    This map is designed to train you on using a rocket launcher.
    It is a rectangular map with a monster on the opposite side. You need
    to use your rocket launcher to kill it. The rocket adds a delay between
    the moment it is fired and the moment it reaches the other side of the room.
    You need to predict the position of the monster to kill it.

    Allowed actions:
        [0]  - ATTACK                           - Shoot weapon - Values 0 or 1
        [14] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [15] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Killing the monster
        -0.0001 - 35 times per second - Kill the monster faster!

    Goal: 0.5 point
        Kill the monster

    Hint: Missile launcher takes longer to load. You must wait a good second after the game starts
        before trying to fire it.

    Ends when:
        - Monster is dead
        - Out of missile (you only have one)
        - Timeout (20 seconds - 700 frames)

    Actions:
        actions = [0] * 43
        actions[0] = 0       # ATTACK
        actions[14] = 1      # TURN_RIGHT
        actions[15] = 0      # TURN_LEFT

    Configuration:
        After creating the env, you can call env.configure() to configure some parameters.

        - lock [e.g. env.configure(lock=multiprocessing_lock)]

            VizDoom requires a multiprocessing lock when running across multiple processes, otherwise the vizdoom instance
            might crash on launch

            You can either:

            1) [Preferred] Create a multiprocessing.Lock() and pass it as a parameter to the configure() method
                [e.g. env.configure(lock=multiprocessing_lock)]

            2) Create and close a Doom environment before running your multiprocessing routine, this will create
                a singleton lock that will be cached in memory, and be used by all Doom environments afterwards
                [e.g. env = gym.make('Doom-...'); env.close()]

            3) Manually wrap calls to reset() and close() in a multiprocessing.Lock()

    Wrappers:

        You can use wrappers to further customize the environment

            import gym
            from gym.wrappers import doom
            env = doom.WrapperTwo(doom.WrapperOne(gym.make('DoomPredictPosition-v0'))

        - Observation space:

            You can use the following wrappers to change the screen resolution

            'Res160x120', 'Res200x125', 'Res200x150', 'Res256x144', 'Res256x160', 'Res256x192', 'Res320x180', 'Res320x200',
            'Res320x240', 'Res320x256', 'Res400x225', 'Res400x250', 'Res400x300', 'Res512x288', 'Res512x320', 'Res512x384',
            'Res640x360', 'Res640x400', 'Res640x480', 'Res800x450', 'Res800x500', 'Res800x600', 'Res1024x576', 'Res1024x640',
            'Res1024x768', 'Res1280x720', 'Res1280x800', 'Res1280x960', 'Res1280x1024', '1400x787', 'Res1400x875',
            'Res1400x1050', 'Res1600x900', 'Res1600x1000', 'Res1600x1200', 'Res1920x1080'

        - Action space:

            'DiscreteMinimal' - Discrete action space with NOOP and only the level's allowed actions
            'Discrete7'       - Discrete action space with NOOP + 7 minimum actions required to complete all levels
            'Discrete17'      - Discrete action space with NOOP + 17 most common actions
            'DiscreteFull'    - Discrete action space with all available actions (Deltas will not work, since their range is restricted)

            'BoxMinimal'      - Box action space with only the level's allowed actions
            'Box7'            - Box action space with 7 minimum actions required to complete all levels
            'Box17'           - Box action space with 17 most common actions
            'BoxFull'         - Box action space with all available actions

            Note: Discrete action spaces only allow one action at a time, Box action spaces support simultaneous actions

        - Control:

            'HumanPlayer' - Use this wrapper if you want to play the game yourself or look around a certain level
                            (controls are Arrow Keys, '<', '>' and Ctrl)
            'Skip1'       - Sends action and repeats it for 1 additional step   (~ 18 fps)
            'Skip2'       - Sends action and repeats it for 2 additional steps  (~ 12 fps)
            'Skip3'       - Sends action and repeats it for 3 additional steps  (~ 9 fps)
            'Skip4'       - Sends action and repeats it for 4 additional steps  (~ 7 fps)
            'Skip5'       - Sends action and repeats it for 5 additional steps  (~ 6 fps)

    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomPredictPositionEnv, self).__init__(6)
