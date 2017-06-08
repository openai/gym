import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import copy

from gym.envs.dart.parameter_managers import *
import copy

import joblib, os


class DartHopperEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = 200
        self.train_UP = False
        self.noisy_input = False
        self.resample_MP = False  # whether to resample the model paraeters
        obs_dim = 11
        self.param_manager = hopperContactMassManager(self)
        modelpath = os.path.join(os.path.dirname(__file__), "models")
        upselector = joblib.load(os.path.join(modelpath, 'UPSelector_torsostrength_sd5_cont.pkl'))
        self.param_manager.sampling_selector = upselector
        self.param_manager.selector_target = 2
        
        if self.train_UP:
            obs_dim += self.param_manager.param_dim

        dart_env.DartEnv.__init__(self, 'hopper_capsule.skel', 4, obs_dim, self.control_bounds, disableViewer=True)

        self.dart_world.set_collision_detector(3) # 3 is ode collision detector
        
        '''curcontparam = copy.copy(self.param_manager.controllable_param)
        self.param_manager.controllable_param = [0, 1, 2, 3, 4]
        self.param_manager.set_simulator_parameters([0.5, 0.5, 0.5, 0.5, 0.5])
        self.param_manager.controllable_param = curcontparam'''

        utils.EzPickle.__init__(self)


    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        pre_state = [self.state_vector()]
        if self.train_UP:
            pre_state.append(self.param_manager.get_simulator_parameters())
        posbefore = self.robot_skeleton.q[0]
        self.advance(a)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        alive_bonus = 1.0
        reward = 0.6*(posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= 5e-1 * joint_limit_penalty
        #reward -= 1e-7 * total_force_mag
        #print(abs(ang))
        div = self.get_div()
        reward -= 1e-1 * np.min([(div**2), 10])
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (height < 1.8) and (abs(ang) < .4))
        ob = self._get_obs()

        return ob, reward, done, {'model_parameters':self.param_manager.get_simulator_parameters(), 'vel_rew':(posafter - posbefore) / self.dt, 'action_rew':1e-3 * np.square(a).sum(), 'forcemag':1e-7*total_force_mag, 'done_return':done}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])
        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))
        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        if self.resample_MP:
            self.param_manager.resample_parameters()
        self.state_action_buffer = [] # for UPOSI

        state = self._get_obs()

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5

    def get_div(self):
        div = 0
        cur_state = self.state_vector()
        d_state0 = self.get_d_state(cur_state)
        dv = 0.001
        for j in [3,4,5,9,10,11]:
            pert_state = np.array(cur_state)
            pert_state[j] += dv
            d_state1 = self.get_d_state(pert_state)

            div += (d_state1[j] - d_state0[j]) / dv
        self.set_state_vector(cur_state)
        return div

    def get_d_state(self, state):
        self.set_state_vector(state)
        self.advance(np.array([0, 0, 0]))
        next_state = self.state_vector()
        d_state = next_state - state
        return d_state