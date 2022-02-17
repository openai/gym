from gym.envs.vizdoom import VizdoomEnv


class VizdoomBasic(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomBasic, self).__init__(0, **kwargs)


class VizdoomCorridor(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomCorridor, self).__init__(1, **kwargs)


class VizdoomDeathmatch(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomDeathmatch, self).__init__(8, **kwargs)


class VizdoomDefendCenter(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomDefendCenter, self).__init__(2, **kwargs)


class VizdoomDefendLine(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomDefendLine, self).__init__(3, **kwargs)


class VizdoomHealthGathering(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomHealthGathering, self).__init__(4, **kwargs)


class VizdoomHealthGatheringSupreme(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomHealthGatheringSupreme, self).__init__(9, **kwargs)


class VizdoomMyWayHome(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomMyWayHome, self).__init__(5, **kwargs)


class VizdoomPredictPosition(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomPredictPosition, self).__init__(6, **kwargs)


class VizdoomTakeCover(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomTakeCover, self).__init__(7, **kwargs)
