__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env

# swing up and balance of double inverted pendulum
class DartDoubleInvertedPendulumEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0],[-1.0]])
        self.action_scale = 40
        dart_env.DartEnv.__init__(self, 'inverted_double_pendulum.skel', 2, 6, control_bounds, dt=0.01)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        reward = 2.0

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = a[0] * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        reward -= 0.01*ob[0]**2
        reward += np.cos(ob[1]) + np.cos(ob[2])
        if (np.cos(ob[1]) + np.cos(ob[2])) > 1.8:
            reward += 5

        notdone = np.isfinite(ob).all() and np.abs(ob[1]) <= np.pi * 3.5 and np.abs(ob[2]) <= np.pi * 3.5# and (np.abs(ob[1]) <= .2) and (np.abs(ob[2]) <= .2)
        done = not notdone
        return ob, reward, done, {}


    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
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
