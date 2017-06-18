__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class DartCartPoleSwingUpEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0],[-1.0]])
        self.action_scale = 40
        dart_env.DartEnv.__init__(self, 'cartpole_swingup.skel', 2, 4, self.control_bounds, dt=0.01)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = a[0] * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        ang = self.robot_skeleton.q[1]

        alive_bonus = 6.0
        ang_cost = 1.0*np.abs(ang)
        quad_ctrl_cost = 0.01 * np.square(a).sum()
        com_cost = 0.01 * np.abs(self.robot_skeleton.q[0])

        reward = alive_bonus - ang_cost - quad_ctrl_cost - com_cost

        done = abs(ang) > 8 * np.pi or abs(self.robot_skeleton.dq[1]) > 25 or abs(self.robot_skeleton.q[0]) > 5

        return ob, reward, bool(done), {}


    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        if self.np_random.uniform(low=0, high=1, size=1) > 0.5:
            qpos[1] += np.pi
        else:
            qpos[1] += -np.pi

        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
