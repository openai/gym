__author__ = 'yuwenhao'

import numpy as np
from gym import utils, spaces
from gym.envs.dart import dart_env


class DartWalker3dSPDEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*15,[-1.0]*15])

        kp_diag = np.array([0.0] * 6 + [300.0] * (15))
        kp_diag[0:3] = 1000
        kp_diag[7:9] = 100
        kp_diag[13:15] = 100
        self.Kp = np.diagflat(kp_diag)
        self.Kd = np.diagflat(kp_diag / 10.0)

        self.torque_limit = np.array([200] * 15)
        self.torque_limit[[-1,-2,-7,-8]] = 20
        self.torque_limit[[0, 1, 2]] = 100

        obs_dim = 42

        self.t = 0

        dart_env.DartEnv.__init__(self, 'walker3d_waist.skel', 4, obs_dim, self.control_bounds, disableViewer=False)

        utils.EzPickle.__init__(self)

    def _spd(self, target_q):
        invM = np.linalg.inv(self.robot_skeleton.M + self.Kd * self.dt)
        p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.dt - target_q)
        d = -self.Kd.dot(self.robot_skeleton.dq)
        qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.dt

        tau[0:6] = 0

        for i in range(len(self.torque_limit)):
            if abs(tau[i+6]) > self.torque_limit[i]:
                tau[i+6] = np.sign(tau[i+6]) * self.torque_limit[i]

        return tau

    def do_simulation_spd(self, target, n_frames):
        for i in range(n_frames):
            tau = self._spd(target)
            self.do_simulation(tau, 1)

    def _step(self, a):
        a[1:3] *= -1
        a[5] *= -1
        pre_state = [self.state_vector()]

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        target_q = np.zeros(self.robot_skeleton.ndofs)
        for i in range(len(self.control_bounds[0])):
            target_q[6 + i] = (clamped_control[i] + 1.0) / 2.0 * (self.robot_skeleton.q_upper[i+6] - self.robot_skeleton.q_lower[i+6]) + self.robot_skeleton.q_lower[i+6]

        posbefore = self.robot_skeleton.bodynodes[0].com()[0]
        self.do_simulation_spd(target_q, self.frame_skip)
        posafter = self.robot_skeleton.bodynodes[0].com()[0]
        height = self.robot_skeleton.bodynodes[0].com()[1]
        side_deviation = self.robot_skeleton.bodynodes[0].com()[2]

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynodes[0].to_world(np.array([0, 1, 0])) - self.robot_skeleton.bodynodes[0].to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynodes[0].to_world(np.array([1, 0, 0])) - self.robot_skeleton.bodynodes[0].to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)
        ang_cos_fwd = np.arccos(ang_cos_fwd)

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
        vel_rew = 0.25 * (posafter - posbefore) / self.dt
        action_pen = 1e-4 * np.square(a).sum()
        joint_pen = 5e-1 * joint_limit_penalty
        deviation_pen = 1e-2 * abs(side_deviation)
        reward = vel_rew + alive_bonus - action_pen - joint_pen - deviation_pen
        #reward -= 1e-7 * total_force_mag

        self.t += self.dt

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > 1.05) and (height < 2.0) and (abs(ang_cos_uwd) < 0.54) and (abs(ang_cos_fwd) < 0.54))

        ob = self._get_obs()

        return ob, reward, done, {'pre_state':pre_state, 'vel_rew':vel_rew, 'action_pen':action_pen, 'joint_pen':joint_pen, 'deviation_pen':deviation_pen, 'done_return':done}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[0:3],
            self.robot_skeleton.q[4:],
            np.clip(self.robot_skeleton.dq,-10,10),
            [self.t]
        ])
        state[3] = self.robot_skeleton.bodynodes[0].com()[1]


        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        self.t = 0

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -5.5
