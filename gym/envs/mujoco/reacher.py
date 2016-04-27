import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        self.finalize()

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid=0

    def _reset(self):
        qpos = np.random.uniform(low=-0.1, high=0.1, size=(self.model.nq,1)) + self.init_qpos
        while True:
            self.goal = np.random.uniform(low=-.2, high=.2, size=(2,1))
            if np.linalg.norm(self.goal) < 2: break
        qpos[-2:] = self.goal
        self.model.data.qpos = qpos
        qvel = self.init_qvel + np.random.rand(self.model.nv,1)*.01-.005
        qvel[-2:] = 0
        self.model.data.qvel = qvel
        self.reset_viewer_if_necessary()
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
