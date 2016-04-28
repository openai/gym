import os.path

import numpy as np
import gym
from gym import error, spaces

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

BIG=10000

class MujocoEnv(gym.Env):
    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist"%fullpath)
        self.frame_skip= frame_skip
        self.model = mujoco_py.MjModel(fullpath)
        self.data = self.model.data
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : int(np.round(1.0 / self.dt))
        }

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.model.data.ctrl = ctrl
        for _ in range(n_frames):
            self.model.step()

    def finalize(self):
        self.init_qpos = self.model.data.qpos.copy()
        self.init_qvel = self.model.data.qvel.copy()
        self.ctrl_dim = self.model.data.ctrl.size
        observation, _reward, done, _info = self.step(np.zeros(self.ctrl_dim))
        assert not done
        self.obs_dim = observation.size

        high = np.ones(self.ctrl_dim)
        low = -high
        self.action_space = spaces.Box(low, high)

        high = BIG*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().finish()
            return

        if mode == 'rgb_array':
            self._get_viewer().render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        elif mode is 'human':
            self._get_viewer().loop_once()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def viewer_setup(self):
        pass

    def reset_viewer_if_necessary(self):
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewer_setup()

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.body_comvels[idx]

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    @property
    def action_bounds(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return lb, ub

    @property
    def _state(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])
