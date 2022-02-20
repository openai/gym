from gym.envs.vizdoom import VizdoomEnv


class VizdoomBasic(VizdoomEnv):
    """
    ### Description

    The purpose of the scenario is just to check if using this framework to train some AI in 3D environment is feasible.
    Map is a rectangle with gray walls, ceiling and floor. Player is spawned along the longer wall, in the center. A red,
    circular monster is spawned randomly somewhere along the opposite wall. Player can only go left/right and
    shoot. 1 hit is enough to kill the monster. Episode finishes when monster is killed or on timeout.

    ### Rewards
    +101 for killing the monster
    -5 for missing
    Episode ends after killing the monster or on timeout.
    -1 for living
    """

    def __init__(self, **kwargs):
        super(VizdoomBasic, self).__init__(0, **kwargs)


class VizdoomCorridor(VizdoomEnv):
    """
    ### Description

    The purpose of this scenario is to teach the agent to navigate towards its fundamental goal (the vest) and make sure
    it survives at the same time. Map is a corridor with shooting monsters on both sides (6 monsters in total). A green
    vest is placed at the opposite end of the corridor. Reward is proportional (negative or positive) to change in the
    distance between the player and the vest. If player ignores monsters on the sides and runs straight for the vest it
    will be killed somewhere along the way.

    ### Rewards
    +dX for getting closer to the vest.
    -dX for getting further from the vest.
    -100 for dying
    """

    def __init__(self, **kwargs):
        super(VizdoomCorridor, self).__init__(1, **kwargs)


class VizdoomDeathmatch(VizdoomEnv):
    """
    ### Description

    Player is spawned along the longer wall, in the center. A red, circular monster is spawned randomly somewhere along the
    opposite wall. Player can only (config) go left/right and shoot. 1 hit is enough to kill the monster. Episode finishes when
    monster is killed or on timeout.

    """

    def __init__(self, **kwargs):
        super(VizdoomDeathmatch, self).__init__(8, **kwargs)


class VizdoomDefendCenter(VizdoomEnv):
    """
    ### Description

    The purpose of this scenario is to teach the agent that killing the monsters is good and being killed by monsters is bad.
    In addition, wasting ammunition is not very good either. Agent is rewarded only for killing monsters so it has to figure
    out the rest for itself. Map is a rectangle. Player is spawned along the longer wall, in the center. 3 melee-only and
    3 shooting monsters are spawned along the opposite wall. Monsters are killed after a single shot, at first. After
    dying each monster is respawned after some time and can endure more damage. Episode ends when the player dies (it's
    inevitable because of limited ammo).

    ### Rewards
    +1 for killing a monster
    -1 for dying
    """

    def __init__(self, **kwargs):
        super(VizdoomDefendCenter, self).__init__(2, **kwargs)


class VizdoomDefendLine(VizdoomEnv):
    """
    ### Description

    The purpose of the scenario is just to check if using this framework to train some AI in 3D environment is feasible.
    Map is a rectangle with gray walls, ceiling and floor. Player is spawned along the longer wall, in the center. A red,
    circular monster is spawned randomly somewhere along the opposite wall. Player can only (config) go left/right and
    shoot. 1 hit is enough to kill the monster. Episode finishes when monster is killed or on timeout.

     ### Rewards
     +1 for killing a monster
     -1 for dying
    """

    def __init__(self, **kwargs):
        super(VizdoomDefendLine, self).__init__(3, **kwargs)


class VizdoomHealthGathering(VizdoomEnv):
    """
    ### Description

    The purpose of this scenario is to teach the agent how to survive without knowing what makes it survive. Agent knows
    only that life is precious and death is bad so it must learn what prolongs its existence and that its health is
    connected with it. Map is a rectangle with green, acidic floor which hurts the player periodically. Initially there are
    some medkits spread uniformly over the map. A new medkit falls from the skies every now and then. Medkits heal some
    portions of player's health - to survive agent needs to pick them up. Episode finishes after player's death or on
    timeout.

    ### Rewards
    living_reward = 1
    death penalty = 100
    """

    def __init__(self, **kwargs):
        super(VizdoomHealthGathering, self).__init__(4, **kwargs)


class VizdoomHealthGatheringSupreme(VizdoomEnv):
    """
    ### Description

    The purpose of this scenario is to teach the agent how to survive without knowing what makes it survive. Agent knows
    only that life is precious and death is bad so it must learn what prolongs its existence and that its health is
    connected with it. Map is a rectangle with green, acidic floor which hurts the player periodically. Initially there are
    some medkits spread uniformly over the map. A new medkit falls from the skies every now and then. Medkits heal some
    portions of player's health - to survive agent needs to pick them up. Episode finishes after player's death or on
    timeout.

    ### Rewards
    living_reward = 1
    death penalty = 100
    """

    def __init__(self, **kwargs):
        super(VizdoomHealthGatheringSupreme, self).__init__(9, **kwargs)


class VizdoomMyWayHome(VizdoomEnv):
    """
    ### Description

    The purpose of this scenario is to teach the agent how to navigate in a labirynth-like surroundings and reach its
    ultimate goal (and learn what it actually is). Map is a series of rooms with interconnection and 1 corridor with a dead
    end. Each room has a different color. There is a green vest in one of the rooms (the same room every time). Player is
    spawned in randomly chosen room facing a random direction. Episode ends when vest is reached or on timeout.

    ### Rewards
    +1 for reaching the vest
    living reward = -0.0001
    """

    def __init__(self, **kwargs):
        super(VizdoomMyWayHome, self).__init__(5, **kwargs)


class VizdoomPredictPosition(VizdoomEnv):
    """
    ### Description

    The purpose of the scenario is teach agent to synchronize missile weapon shot (involving a signifficant delay between
    shooting and hitting) with target movements. Agent should be able to shoot so that missile and monster meet each other.
    The map is a rectangle room. Player is spawned along the longer wall, in the center. A monster is spawned randomly
    somewhere along the opposite wall and walks between left and right corners along the wall. Player is equipped with a
    rocket launcher and a single rocket. Episode ends when missile hits a wall/the monster or on timeout.

    ### Rewards
    +1 for killing the monster
    """

    def __init__(self, **kwargs):
        super(VizdoomPredictPosition, self).__init__(6, **kwargs)


class VizdoomTakeCover(VizdoomEnv):
    """
    ### Description

    The purpose of this scenario is to teach agent to link incoming missles with its estimated lifespan. Agent should
    learn that being hit means health decrease and this in turn will lead to death which is undesirable. In effect agent
    should avoid missles. Map is a rectangle. Player is spawned along the longer wall, in the center. A couple of shooting
    monsters are spawned randomly somewhere along the opposite wall and try to kill the player with fireballs. The player
    can only (config) move left/right. More monsters appear with time. Episode ends when player dies.

    ### Rewards
    +1 for each tic of life
    """

    def __init__(self, **kwargs):
        super(VizdoomTakeCover, self).__init__(7, **kwargs)
