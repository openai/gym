#TODO preprocesing of the observation to match Deepmind's Nature paper

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env_pixel
from skimage import color
from skimage import transform


class HalfCheetahEnvPixel(mujoco_env_pixel.MujocoEnvPixel, utils.EzPickle):
    def __init__(self):
        # self.memory = np.zeros([4,84,84])
        self.memory = np.empty([84,84,4],dtype=np.uint8)
        mujoco_env_pixel.MujocoEnvPixel.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        

    def _step(self, action):
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        data = self._get_viewer().get_image()
        rawByteImg = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(rawByteImg, dtype=np.uint8)
        img = np.reshape(tmp, [height, width, 3])
        img = np.flipud(img) # 500x500x3
        gray = color.rgb2gray(img) # convert to gray (now become 0-1)
        gray_resized = transform.resize(gray,(84,84)) # resize
        # update memory buffer
        # self.memory[1:,:,:] = self.memory[0:3,:,:]
        self.memory[:,:,1:] = self.memory[:,:,0:3]
        # self.memory[0,:,:] = gray_resized
        self.memory[:,:,0] = gray_resized*255

        return self.memory

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


# def get_image(self):
    """
    returns a tuple (data, width, height), where:
    - data is a string with raw bytes representing the pixels in 3-channel RGB
      (i.e. every three bytes = 1 pixel)
    - width is the width of the image
    - height is the height of the image
    """
    # glfw.make_context_current(self.window)
    # width, height = self.get_dimensions()
    # gl.glReadBuffer(gl.GL_BACK)
    # data = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    # return (data, width, height)