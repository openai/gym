from gym import utils
from gym.envs.robotics import fetch_env


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self):
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
        fetch_env.FetchEnv.__init__(
            self, 'fetch/pick_and_place.xml', has_box=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_x_shift=0.0,
            obj_range=0.15, target_range=0.15, dist_threshold=0.05, initial_qpos=initial_qpos)
        utils.EzPickle.__init__(self)
