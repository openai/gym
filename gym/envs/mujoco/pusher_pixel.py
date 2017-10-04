import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env_pixel

import mujoco_py
from mujoco_py.mjlib import mjlib

from skimage import color
from skimage import transform

class PusherEnvPixel(mujoco_env_pixel.MujocoEnvPixel, utils.EzPickle):
    def __init__(self):
        self.memory = np.empty([84,84,4],dtype=np.uint8)
        utils.EzPickle.__init__(self)
        mujoco_env_pixel.MujocoEnvPixel.__init__(self, 'pusher.xml', 5)

    def _step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        data = self._get_viewer().get_image()
        rawByteImg = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(rawByteImg, dtype=np.uint8)
        img = np.reshape(tmp, [height, width, 3])
        img = np.flipud(img) # 500x500x3
        gray = color.rgb2gray(img) # convert to gray
        gray_resized = transform.resize(gray,(84,84)) # resize
        # update memory buffer
        # self.memory[1:,:,:] = self.memory[0:3,:,:]
        self.memory[:,:,1:] = self.memory[:,:,0:3]
        # self.memory[0,:,:] = gray_resized
        self.memory[:,:,0] = gray_resized*255

        return self.memory
