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
        fetch_env.FetchEnv.__init__(
            self, 'reach.xml', has_box=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_x_shift=0.0,
            obj_range=0.15, target_range=0.15)
        utils.EzPickle.__init__(self)
