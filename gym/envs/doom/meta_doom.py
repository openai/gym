"""
------------        Meta - Doom         ------------
This is a meta map that combines all 9 Doom levels.

Levels:

    0   - Doom Basic
    1   - Doom Corridor
    2   - Doom DefendCenter
    3   - Doom DefendLine
    4   - Doom HealthGathering
    5   - Doom MyWayHome
    6   - Doom PredictPosition
    7   - Doom TakeCover
    8   - Doom Deathmatch

Goal: 9,000 points
    - Pass all levels

Scoring:
    - Each level score has been standardized on a scale of 0 to 1,000
    - The passing score for a level is 990 (99th percentile)
    - A bonus of 450 (50 * 9 levels) is given if all levels are passed
    - The score for a level is the average of the last 3 tries
    - If there has been less than 3 tries for a level, the missing tries will have a score of 0
      (e.g. if you score 1,000 on the first level on your first try, your level score will be (1,000+0+0)/ 3 = 333.33)
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
        - you can see
            the current level with info["LEVEL"]
            the list of level score with info["SCORES"],
            the list of locked levels with info["LOCKED_LEVELS"]
            your total score with info["TOTAL_REWARD"]

        e.g.
            import gym
            env = gym.make('meta-Doom-v0')
            env.reset()
            total_score = 0
            while total_score < 9000:
                action = [0] * 43
                obs, reward, is_finished, info = env.step(action)
                env.render()
                total_score = info["TOTAL_REWARD"]
                if is_finished:
                    env.change_level(level_you_want)

    2) Automatic change

        - if you don't call change_level() and the level is finished, the system will automatically select the
          unlocked level with the lowest level score (which is likely to be the last unlocked level)

        e.g.
            import gym
            env = gym.make('meta-Doom-v0')
            env.reset()
            total_score = 0
            while total_score < 9000:
                action = [0] * 43
                obs, reward, is_finished, info = env.step(action)
                env.render()
                total_score = info["TOTAL_REWARD"]

Allowed actions:
    - Each level has their own allowed actions, see each level for details

Actions:
    actions = [0] * 43
    actions[0] = 0       # ATTACK
    actions[1] = 0       # USE
    [...]
    actions[42] = 0      # MOVE_UP_DOWN_DELTA
    A full list of possible actions is available in controls.md

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
            env = doom.WrapperTwo(doom.WrapperOne(gym.make('meta-Doom-v0'))

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