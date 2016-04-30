import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)
        self.finalize()

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self._state
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > 0.7) and (abs(ang) < 0.2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    def _reset(self):
        self.model.data.qpos = (self.init_qpos +
                                np.random.rand(self.model.nq, 1) * 0.01 -
                                0.005)
        self.model.data.qvel = (self.init_qvel +
                                np.random.rand(self.model.nv, 1) * 0.01 -
                                0.005)
        self.reset_viewer_if_necessary()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += 0.8
        self.viewer.cam.elevation = -20
