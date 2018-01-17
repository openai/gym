import numpy as np
from gym import utils
from gym.envs.fetch import fetch_env


class FetchKickEnv(fetch_env.FetchEnv, utils.EzPickle):
    initial_qpos = {
        'robot0:slide0': 0.05,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'geom0:slide0': 0.7,
        'geom0:slide1': 0.3,
        'geom0:slide2': 0.0,
        'geom1:slide0': 1.703020558521492,
        'geom1:slide1': 1.0816411287521643,
        'geom1:slide2': 0.4,
    }

    def __init__(self):
        fetch_env.FetchEnv.__init__(self, 'kick.xml', n_boxes=1)
        utils.EzPickle.__init__(self)
