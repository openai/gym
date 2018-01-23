import numpy as np
from gym import utils
from gym.envs.fetch import fetch_env


class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
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
        fetch_env.FetchEnv.__init__(
            self, 'slide.xml', has_box=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_x_shift=0.4,
            obj_range=0.1, target_range=0.3, dist_threshold=0.05)
        utils.EzPickle.__init__(self)
