try:
    import Box2D
    from gym.envs.box2d.lunar_lander import LunarLander, LunarLanderContinuous
    from gym.envs.box2d.bipedal_walker import BipedalWalker, BipedalWalkerHardcore
    from gym.envs.box2d.car_racing import CarRacing
except ImportError:
    Box2D = None
