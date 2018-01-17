import numpy as np
from gym import utils
from gym.envs.fetch import fetch_env
from gym import spaces


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self):
        fetch_env.FetchEnv.__init__(self, 'reach.xml', 4)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        return 0, 0, 0, {}

    def _reset(self):
        pass

    @property
    def observation_space(self):
        return spaces.Box(-np.ones(2), np.ones(2))

    @property
    def initial_qpos(self):
        init_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table_table20:slide0': 1.05,
            'table_table20:slide1': 0.4,
            'table_table20:slide2': 0.0,
        }
        return init_qpos

    def initial_setup(self):
        fetch_env.FetchEnv.initial_setup(self)
