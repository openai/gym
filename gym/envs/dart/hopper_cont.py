__author__ = 'yuwenhao'

import numpy as np
from gym import utils, spaces
from gym.envs.dart import dart_env

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import Layer
import theano.tensor as T, theano
from keras import backend as K
import numpy as np
import copy
import os

import joblib

# WARNING: A lot of hand-coded stuff for now
class DartHopperEnvCont(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = 200
        self.train_UP = False
        self.noisy_input = False
        obs_dim = 11

        # UPOSI variables
        self.use_OSI = False
        self.history_length = 5 # size of the motion history for UPOSI
        self.state_action_buffer = []

        modelpath = os.path.join(os.path.dirname(__file__), "models")
        self.UP = joblib.load(os.path.join(modelpath, 'UP.pkl'))
        if self.use_OSI:
            self.OSI = load_model(os.path.join(modelpath, 'OSI.h5'))
            self.OSI_out = K.function([self.OSI.input, K.learning_phase()], self.OSI.output)

        dart_env.DartEnv.__init__(self, 'hopper_capsule.skel', 4, obs_dim, self.control_bounds)

        if self.use_OSI:
            self.OSI_obs_dim = (self.obs_dim+self.act_dim)*self.history_length+self.obs_dim
            self.obs_dim = 2
            self.observation_space = spaces.Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        self.act_dim = 2
        self.action_space = spaces.Box(np.array([0, 0]), np.array([1, 1]))

        utils.EzPickle.__init__(self)

    def _step(self, a):
        if not len(a) == 2:
            action = [0, 0, 0]
        elif self.use_OSI:
            cur_obs = np.hstack([self.state_action_buffer[-1][0], a])
            _, data = self.UP.get_action(cur_obs)
            action = data['mean']
            self.state_action_buffer[-1].append(np.array(action))
        else:
            cur_obs = np.hstack([self._get_obs(), a])
            _, data = self.UP.get_action(cur_obs)
            action = data['mean']

        clamped_control = np.array(action)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        posbefore = self.robot_skeleton.q[0]
        self.do_simulation(tau, self.frame_skip)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        joint_limit_penalty = 0
        for j in range(3, self.robot_skeleton.ndofs):
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.0)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.0)

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (height < 1.8) and (abs(ang) < .4))
        ob = self._get_obs()

        return ob, reward, done, {'vel_rew':(posafter - posbefore) / self.dt, 'action_rew':1e-3 * np.square(action).sum()}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if not self.use_OSI:
            return state

        out_ob = np.zeros(self.OSI_obs_dim)
        ind = 0
        for s_a in self.state_action_buffer:
            out_ob[ind:ind+len(s_a[0])] = np.array(s_a[0])
            ind += len(s_a[0])
            out_ob[ind:ind+len(s_a[1])] = np.array(s_a[1])
            ind += len(s_a[1])
        out_ob[ind:ind + len(state)] = np.array(state)

        self.state_action_buffer.append([np.array(state)])
        if len(self.state_action_buffer) > self.history_length:
            self.state_action_buffer.pop(0)
        pred_param = self.OSI_out([[out_ob], 0])[0]

        return pred_param

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        if self.train_UP:
            self.param_manager.resample_parameters()

        self.state_action_buffer = [] # for UPOSI

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5