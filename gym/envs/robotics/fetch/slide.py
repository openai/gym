import numpy as np

from gym import utils
from gym.envs.robotics import fetch_env


class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 0.7,
            'table0:slide1': 0.3,
            'table0:slide2': 0.0,
            'object0:slide0': 1.703020558521492,
            'object0:slide1': 1.0816411287521643,
            'object0:slide2': 0.4,
        }
        fetch_env.FetchEnv.__init__(
            self, 'fetch/slide.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
