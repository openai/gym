# This environment is created by Dong Xu (donghsu@gatech.edu)

import numpy as np
from gym import utils
from gym import spaces
from gym.envs.dart import dart_env

class DartCartPoleImgEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.x_threshold = 1.4
        self.pole_theta_threshold = 0.268
        self.cart_pos_x = 0.0
        self.pole_rotate = 0.0
        self.cart_pos_x_old = 0.0
        self.pole_rotate_old = 0.0
        self.cart_spd = 0.0
        self.pole_spd = 0.0

        self.screen_width = 80
        self.screen_height = 45
        control_bounds = np.array([[1.0],[-1.0]])
        self.action_space = spaces.Discrete(2)
        dart_env.DartEnv.__init__(self, 'cartpole.skel', 2, 4, control_bounds, \
                                  obs_type="image", action_type="discrete", visualize=False, \
                                  screen_width=self.screen_width, screen_height=self.screen_height)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        tau = np.zeros(self.robot_skeleton.ndofs)
        if a == 0:
            tau[0] = -10
        else:
            tau[0] = 10

        self.cart_pos_x  = self.robot_skeleton.body('cart').transform()[0][3]
        self.pole_rotate = self.robot_skeleton.body('pole').transform()[0][1]
        self.cart_spd = (self.cart_pos_x - self.cart_pos_x_old) / self.dt
        self.pole_spd = (self.pole_rotate - self.pole_rotate_old) / self.dt
        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        done = self.cart_pos_x  < -self.x_threshold \
            or self.cart_pos_x  >  self.x_threshold \
            or self.pole_rotate < -self.pole_theta_threshold \
            or self.pole_rotate >  self.pole_theta_threshold
        done = bool(done)

        reward = 1.0
        if done:
            reward = 0.0
        
        return ob, reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            data = self._get_viewer().getFrame()
            return data
        elif mode == 'human':
            self._get_viewer().runSingleStep()

    def _get_obs(self):
        return self._get_viewer().getGrayscale(self.screen_width, self.screen_height)

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[1] = -0.2
        self._get_viewer().scene.tb.trans[2] = -1.4
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
