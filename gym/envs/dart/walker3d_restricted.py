__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartWalker3dRestrictedEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*15,[-1.0]*15])
        self.action_scale = np.array([100.0]*15)
        self.action_scale[[-1,-2,-7,-8]] = 20
        self.action_scale[[0, 1, 2]] = 150
        obs_dim = 41

        self.t = 0

        dart_env.DartEnv.__init__(self, 'walker3d_waist_restricted.skel', 8, obs_dim, self.control_bounds, disableViewer=False)

        self.robot_skeleton.set_self_collision_check(True)

        for i in range(1, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(0)

        utils.EzPickle.__init__(self)

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        #self.do_simulation(tau*0, self.frame_skip)

    def _step(self, a):
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.bodynodes[0].com()[0]
        self.advance(a)

        posafter = self.robot_skeleton.bodynodes[0].com()[0]
        height = self.robot_skeleton.bodynodes[0].com()[1]
        side_deviation = self.robot_skeleton.bodynodes[0].com()[2]

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynodes[1].to_world(np.array([0, 1, 0])) - self.robot_skeleton.bodynodes[1].to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynodes[1].to_world(np.array([1, 0, 0])) - self.robot_skeleton.bodynodes[1].to_world(np.array([0, 0, 0]))
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
        vel_rew = 2.0 * (posafter - posbefore) / self.dt
        action_pen = 1e-2 * np.square(a).sum()
        joint_pen = 0 * joint_limit_penalty
        deviation_pen = 1e-3 * abs(side_deviation)
        reward = vel_rew + alive_bonus - action_pen - joint_pen - deviation_pen

        action_vio = np.sum(np.exp(np.max([(a-self.control_bounds[0]), [0]*15], axis=0)) - [1]*15)
        action_vio += np.sum(np.exp(np.max([(self.control_bounds[1]-a), [0]*15], axis=0)) - [1]*15)
        reward -= 0.1*action_vio

        #reward -= 1e-7 * total_force_mag

        #div = self.get_div()
        #reward -= 1e-1 * np.min([(div**2), 10])

        self.t += self.dt

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > 1.05) and (height < 2.0) and (abs(ang_cos_uwd) < 0.84) and (abs(ang_cos_fwd) < 0.84))

        '''if (np.max([(a-self.control_bounds[0]), [0]*15], axis=0) > 0.1).any():
            done = True
        if (np.max([(self.control_bounds[1] - a), [0]*15], axis=0) > 0.1).any():
            done = True'''


        if done:
            reward = 0

        ob = self._get_obs()

        foot1_com = self.robot_skeleton.bodynode('h_foot').com()
        foot2_com = self.robot_skeleton.bodynode('h_foot_left').com()
        robot_com = self.robot_skeleton.com()
        com_foot_offset1 = robot_com - foot1_com
        com_foot_offset2 = robot_com - foot2_com

        return ob, reward, done, {'pre_state':pre_state, 'vel_rew':vel_rew, 'action_pen':action_pen, 'joint_pen':joint_pen, 'deviation_pen':deviation_pen, 'aux_pred':np.hstack([com_foot_offset1, com_foot_offset2, [reward]]), 'done_return':done}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10),
            #[self.t]
        ])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        sign = np.sign(np.random.uniform(-1, 1))
        #qpos[9] = sign * self.np_random.uniform(low=0.3, high=0.35, size=1)
        #qpos[15] = -sign * self.np_random.uniform(low=0.3, high=0.35, size=1)
        self.set_state(qpos, qvel)
        self.t = 0

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -5.5

    def get_div(self):
        div = 0
        cur_state = self.state_vector()
        d_state0 = self.get_d_state(cur_state)
        dv = 0.0001
        for j in [6,7,8,12, 18, 27, 28, 29, 33, 39]:
            pert_state = np.array(cur_state)
            pert_state[j] += dv
            d_state1 = self.get_d_state(pert_state)

            div += (d_state1[j] - d_state0[j]) / dv
        self.set_state_vector(cur_state)
        return div

    def get_d_state(self, state):
        self.set_state_vector(state)
        self.advance(np.array([0]*15))
        next_state = self.state_vector()
        d_state = next_state - state
        return d_state