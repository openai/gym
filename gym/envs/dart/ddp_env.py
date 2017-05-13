__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.inverted_double_pendulum import DartDoubleInvertedPendulumEnv
from gym.envs.dart.cartpole_swingup import DartCartPoleSwingUpEnv
from gym.envs.dart.cart_pole import DartCartPoleEnv
from gym.envs.dart import pylqr

class DDPReward:
    def __init__(self):
        self.ilqr_reward_parameters = np.array([0]*4)
        self.coefs = [1, 10, 1]

    def __call__(self, x, u, t, aux):
        '''pos_pen = self.ilqr_reward_parameters[4] * abs((x[0] - self.ilqr_reward_parameters[0]))
        pos_pen += self.ilqr_reward_parameters[5] * abs((x[1] - self.ilqr_reward_parameters[1]))
        vel_pen = self.ilqr_reward_parameters[6] * abs((x[2] - self.ilqr_reward_parameters[2]))
        vel_pen += self.ilqr_reward_parameters[7] * abs((x[3] - self.ilqr_reward_parameters[3]))
        control_pen = self.ilqr_reward_parameters[8] * (np.square(u).sum())'''

        pos_pen = self.coefs[0] * abs((x[0]))
        pos_pen += self.coefs[1] * abs((x[1]))
        control_pen = self.coefs[2] * (np.square(u).sum())

        shaped_reward = self.ilqr_reward_parameters[2] * abs(x[0] - self.ilqr_reward_parameters[0])
        shaped_reward += self.ilqr_reward_parameters[3] * abs(x[1] - self.ilqr_reward_parameters[1])

        return pos_pen + control_pen + shaped_reward

    def cost_dx(self, x, u, t, aux):
        dx = np.zeros(len(x))
        dx[0] = self.coefs[0] * np.sign(x[0]) + self.ilqr_reward_parameters[2] * np.sign(x[0] - self.ilqr_reward_parameters[0])
        dx[1] = self.coefs[1] * np.sign(x[1]) + self.ilqr_reward_parameters[3] * np.sign(x[1] - self.ilqr_reward_parameters[1])
        return dx

    def cost_du(self, x, u, t, aux):
        du = 2 * self.coefs[2] * u
        return du

    def cost_dxx(self, x, u, t, aux):
        return np.zeros((len(x), len(x)))

    def cost_duu(self, x, u, t, aux):
        return np.identity(len(u)) * 2 * self.coefs[2]

    def cost_dux(self, x, u, t, aux):
        return np.zeros((len(u), len(x)))

# swing up and balance of double inverted pendulum
class DDPEnv(DartCartPoleSwingUpEnv, utils.EzPickle):
    def __init__(self):
        self.parentClass = DartCartPoleSwingUpEnv
        self.ddp_horizon = 10
        self.current_step = 0
        self.ddp_reward = DDPReward()
        self.ilqr = pylqr.PyLQR_iLQRSolver(T=self.ddp_horizon, plant_dyn=self.plant_dyn, cost=self.ddp_reward)
        self.ilqr.cost_du = self.ddp_reward.cost_du
        self.ilqr.cost_dx = self.ddp_reward.cost_dx
        self.ilqr.cost_duu = self.ddp_reward.cost_duu
        self.ilqr.cost_dxx = self.ddp_reward.cost_dxx
        self.ilqr.cost_dux = self.ddp_reward.cost_dux

        self.control_bounds = np.array([[2, 6, 10, 10], [-2, -6, 0, 0]])
        self.action_scale = 40
        dart_env.DartEnv.__init__(self, 'cartpole_swingup.skel', 2, 4, self.control_bounds, dt=0.01)

        utils.EzPickle.__init__(self)

    def _step(self, a):
        for i in range(len(a)):
            if a[i] < self.control_bounds[1][i]:
                a[i] = self.control_bounds[1][i]
            if a[i] > self.control_bounds[0][i]:
                a[i] = self.control_bounds[0][i]

        x0 = self.state_vector()
        u_init = np.array([np.array([0]) for t in range(self.ddp_horizon)])
        '''if self.current_step == 0:
            u_init = np.array([np.array([0]) for t in range(self.ddp_horizon)])
        else:
            u_init = self.res['u_array_opt'][1:]
            u_init = np.vstack([u_init, u_init[-1]])'''
        #u_init = np.array([np.array([np.random.uniform(-1, 1)]) for t in range(self.ddp_horizon)])

        self.ddp_reward.ilqr_reward_parameters = np.array(a)

        if self.current_step == 0:
            iter = 15
        else:
            iter = 15

        self.res = self.ilqr.ilqr_iterate(x0, u_init, n_itrs=iter, tol=1e-6, verbose=False)

        self.set_state_vector(x0)

        reward = 0
        #print(self.res['u_array_opt'])
        for u in self.res['u_array_opt']:
            ob, rew, done, info = self.parentClass._step(self, u)
            reward += rew
            if done:
                break
        #ob, reward, done, info = self.parentClass._step(self, self.res['u_array_opt'][0])
        self.current_step += 1
        return ob, reward, done, {}

    def _get_obs(self):
        return self.parentClass._get_obs(self)

    def reset_model(self):
        self.current_step = 0
        return self.parentClass.reset_model(self)

    def viewer_setup(self):
        self.parentClass.viewer_setup(self)

    def plant_dyn(self, x, u, t, aux):
        self.set_state_vector(x)
        self.parentClass._step(self, u)
        x_new = self.state_vector()
        return x_new

