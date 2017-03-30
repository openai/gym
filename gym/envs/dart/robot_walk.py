import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import copy

class DartRobotWalk(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.chase_pos = self.robot_skeleton.com()[0]
        self.chase_vel = 0.01
        self.control_bounds = np.array([[1.0]*16,[-1.0]*16])
        self.action_scale = 0.3
        dart_env.DartEnv.__init__(self, ['ground.urdf', 'BioloidGP/BioloidGP.URDF'], 4, 44, self.control_bounds)
        self.init_q = copy.deepcopy(self.robot_skeleton.q)
        self.init_q[0] = -1.17
        self.init_q[1] = 1.17
        self.init_q[2] = 1.17
        self.init_q[4] += 0.32
        self.robot_skeleton.q = self.init_q
        print(self.robot_skeleton.q)

        utils.EzPickle.__init__(self)

    def in_contact(self, contact, bodynode1, bodynode2):
        if contact.bodynode1 == bodynode1 and contact.bodynode2 == bodynode2:
            return True
        if contact.bodynode2 == bodynode1 and contact.bodynode1 == bodynode2:
            return True
        return False

    def _step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        posbefore = self.robot_skeleton.com()[0]
        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()
        posafter = self.robot_skeleton.com()[0]
        height = self.robot_skeleton.bodynodes[0].com()[1]
        #print(height)

        '''alive_bonus = 0.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        contacts = self.dart_world.collision_result.contacts
        l_incontact = False
        r_incontact = False
        for contact in contacts:
            if self.in_contact(contact, self.dart_world.skeletons[0].bodynodes[1], self.robot_skeleton.bodynode('l_foot')):
                l_incontact = True
            if self.in_contact(contact, self.dart_world.skeletons[0].bodynodes[1], self.robot_skeleton.bodynode('r_foot')):
                r_incontact = True
        if l_incontact and r_incontact:
            reward -= 0.2

        foot_in_range = True
        if self.robot_skeleton.bodynode('l_foot').com()[1] > self.robot_skeleton.bodynode('r_shin').com()[1]:
            foot_in_range = False
        if self.robot_skeleton.bodynode('r_foot').com()[1] > self.robot_skeleton.bodynode('l_shin').com()[1]:
            foot_in_range = False

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynode('torso').to_world(np.array([0, -1, 0])) - self.robot_skeleton.bodynode('torso').to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        reward += np.dot(forward, forward_world)

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .2) and foot_in_range)'''

        alive_bonus = 0.5
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all())
        self.chase_pos += self.chase_vel * self.dt
        if self.chase_pos > posafter:
            done = True

        return ob, reward, done, {}


    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        self.robot_skeleton.q = self.init_q
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        self.chase_pos = self.robot_skeleton.com()[0]
        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -1.5
        self._get_viewer().scene.tb._set_theta(-15)


