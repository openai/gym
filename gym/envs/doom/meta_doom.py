
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
          (e.g. if you score 1,000 on the first level on your first try, your level score will be (1,000 + 0 + 0) / 3 = 333.33)
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
        - To unlock the next level, you must achieve a level score (avg of last 3 tries) of at least 600 (i.e. passing 60% of the last level)
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

    Mode:
        - env.mode can be 'fast', 'normal' or 'human' (e.g. env.mode = 'fast')
        - 'fast' (default) will run as fast as possible (~75 fps) (best for simulation)
        - 'normal' will run at roughly 35 fps (easier for human to watch)
        - 'human' will let you play the game (keyboard: Arrow Keys, '<', '>' and Ctrl, mouse available for Doom Deathmatch)

        e.g. to start in human mode:

            import gym
            env = gym.make('meta-Doom-v0')
            env.mode='human'
            env.reset()
            num_episodes = 10
            for i in range(num_episodes):
                env.step([0] * 43)

    Actions:
        actions = [0] * 43
        actions[0] = 0       # ATTACK
        actions[1] = 0       # USE
        [...]
        actions[42] = 0      # MOVE_UP_DOWN_DELTA
        A full list of possible actions is available in controls.md
    -----------------------------------------------------
    """