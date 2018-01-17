import numpy as np
from gym import utils
from gym.envs.fetch import fetch_env
from gym import spaces


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'table_table20:slide0': 1.05,
        'table_table20:slide1': 0.4,
        'table_table20:slide2': 0.0,
        'geom0:slide0': 1.2517958243065,
        'geom0:slide1': 0.5311251479548121,
        'geom0:slide2': 0.4,
    }

    observation_space = spaces.Box(-np.ones(2), np.ones(2))

    def __init__(self):
        fetch_env.FetchEnv.__init__(self, 'push.xml', n_boxes=1)
        utils.EzPickle.__init__(self)

    def initial_setup(self):
        fetch_env.FetchEnv.initial_setup(self)