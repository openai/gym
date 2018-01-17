import numpy as np
from gym import utils
from gym.envs.fetch import fetch_env


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    initial_qpos = {
        'robot0:slide0': 0.4049,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'table_table20:slide0': 1.05,
        'table_table20:slide1': 0.4,
        'table_table20:slide2': 0.0,
    }

    def __init__(self):
        fetch_env.FetchEnv.__init__(self, 'reach.xml', gripper_extra_height=0.2, n_boxes=0)
        utils.EzPickle.__init__(self)
