import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 4)
        utils.EzPickle.__init__(self)
        self.ctrl_cost_coeff = 0.0001
        self.finalize()

    def _step(self, a):
        xposbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.model.data.qpos[0,0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - self.ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd = reward_fwd, reward_ctrl=reward_ctrl)


    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([
            qpos.flat[2:],
            qvel.flat
        ])

    def _reset(self):
        self.model.data.qpos = self.init_qpos + np.random.uniform(size=(self.model.nq,1),low=-.1,high=.1)
        self.model.data.qvel = self.init_qvel + np.random.uniform(size=(self.model.nv,1),low=-.1,high=.1)
        self.reset_viewer_if_necessary()
        return self._get_obs()
