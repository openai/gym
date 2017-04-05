__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartWalker3dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*12,[-1.0]*12])
        self.action_scale = np.array([100.0]*12)
        self.action_scale[[-1,-2,-7,-8]] = 10
        obs_dim = 35

        dart_env.DartEnv.__init__(self, 'walker3d.skel', 4, obs_dim, self.control_bounds)

        utils.EzPickle.__init__(self)

    def _step(self, a):
        pre_state = [self.state_vector()]

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        posbefore = self.robot_skeleton.bodynodes[0].com()[0]
        self.do_simulation(tau, self.frame_skip)
        posafter = self.robot_skeleton.bodynodes[0].com()[0]
        height = self.robot_skeleton.bodynodes[0].com()[1]
        side_deviation = self.robot_skeleton.bodynodes[0].com()[2]

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynodes[0].to_world(np.array([0, 1, 0])) - self.robot_skeleton.bodynodes[0].to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynodes[0].to_world(np.array([1, 0, 0])) - self.robot_skeleton.bodynodes[0].to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        joint_limit_penalty = 0
        for j in [-3, -9]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-4 * np.square(a).sum()
        reward -= 5e-1 * joint_limit_penalty
        reward -= 1e-2 * abs(side_deviation)
        #reward -= 1e-7 * total_force_mag

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .8) and (height < 2.0) and (ang_cos_uwd > 0.54) and (ang_cos_fwd > 0.54))

        ob = self._get_obs()

        return ob, reward, done, {'pre_state':pre_state, 'vel_rew':(posafter - posbefore) / self.dt, 'action_rew':1e-3 * np.square(a).sum(), 'forcemag':1e-7*total_force_mag, 'done_return':done}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[0:3],
            self.robot_skeleton.q[4:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[3] = self.robot_skeleton.bodynodes[0].com()[1]


        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -5.5
