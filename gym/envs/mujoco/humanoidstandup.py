import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidStandupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoidstandup.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def _step(self, a):
        pos_before = self.model.data.qpos
        self.do_simulation(a, self.frame_skip)
        pos_after = self.model.data.qpos
        data = self.model.data
        qpos = self.model.data.qpos
        uph_cost=(pos_after[2]-pos_before[2]) / self.model.opt.timestep

        quad_ctrl_cost =  0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost

        done = bool((pos_after[1] - pos_before[1] > 15.0) or (pos_after[0] - pos_before[0] > 15.0))
        return self._get_obs(), reward[0], done, dict(reward_linup=uph_cost, reward_quadctrl=-quad_ctrl_cost, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + np.random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + np.random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
