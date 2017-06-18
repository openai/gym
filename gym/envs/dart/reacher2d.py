import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class DartReacher2dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.1, 0.01, -0.1])
        self.action_scale = np.array([200, 200])
        self.control_bounds = np.array([[1.0, 1.0],[-1.0, -1.0]])
        dart_env.DartEnv.__init__(self, 'reacher2d.skel', 2, 11, self.control_bounds, dt=0.01, disableViewer=False)
        for s in self.dart_world.skeletons:
            s.set_self_collision_check(False)
            for n in s.bodynodes:
                n.set_collidable(False)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        vec = self.robot_skeleton.bodynodes[-1].com() - self.target

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()#*0.1
        reward = reward_dist + reward_ctrl

        s = self.state_vector()
        #done = not (np.isfinite(s).all() and (-reward_dist > 0.02))
        done = False

        return ob, reward, done, {}

    def _get_obs(self):
        theta = self.robot_skeleton.q
        vec = self.robot_skeleton.bodynodes[-1].com() - self.target
        return np.concatenate([np.cos(theta), np.sin(theta), [self.target[0], self.target[2]], self.robot_skeleton.dq, vec]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        while True:
            self.target = self.np_random.uniform(low=-.2, high=.2, size=3)
            self.target[1] = 0.0
            if np.linalg.norm(self.target) < .2: break
        self.target[1] = 0.01

        self.dart_world.skeletons[1].q=[0, 0, 0, self.target[0], self.target[1], self.target[2]]


        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -1.0
        self._get_viewer().scene.tb._set_theta(-45)
        self.track_skeleton_id = 0
