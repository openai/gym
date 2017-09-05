import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim, MjViewer
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

def _get_image_hack(self):
    # modified version of _read_pixels_as_in_window
    # waiting on issues mujoco-py#101 and mujoco-py#102
    # Reads pixels with markers and overlay from the same camera as screen.

    import glfw
    import numpy as np
    import copy
    from mujoco_py.utils import rec_copy, rec_assign

    resolution = glfw.get_framebuffer_size(
        self.sim._render_context_window.window)
 
    resolution = np.array(resolution)
    resolution = resolution * min(1000 / np.max(resolution), 1)
    resolution = resolution.astype(np.int32)
    resolution -= resolution % 16
 
    if self.sim._render_context_offscreen is None:
        self.sim.render(resolution[0], resolution[1])
    offscreen_ctx = self.sim._render_context_offscreen
    window_ctx = self.sim._render_context_window
    # Save markers and overlay from offscreen.
    saved = [copy.deepcopy(offscreen_ctx._markers),
             copy.deepcopy(offscreen_ctx._overlay),
             rec_copy(offscreen_ctx.cam)]
    # Copy markers and overlay from window.
    offscreen_ctx._markers[:] = window_ctx._markers[:]
    offscreen_ctx._overlay.clear()
    offscreen_ctx._overlay.update(window_ctx._overlay)
 
    # FIXME
    # rec_assign(offscreen_ctx.cam, rec_copy(window_ctx.cam))
 
    img = self.sim.render(*resolution)
    img = img[::-1, :, :] # Rendered images are upside-down.
    # Restore markers and overlay to offscreen.
    offscreen_ctx._markers[:] = saved[0][:]
    offscreen_ctx._overlay.clear()
    offscreen_ctx._overlay.update(saved[1])
 
    # FIXME
    ## rec_assign(offscreen_ctx.cam, saved[2])
 
    return img

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = load_model_from_path(fullpath)
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange
        # I'm not sure why bounds is at least sometimes None ... bug?
        if bounds is not None:
            bounds = bounds.copy()
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            # self.viewer.autoscale()
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        self.sim.forward()
        # self.model._compute_subtree()  # pylint: disable=W0212

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                # self._get_viewer().finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self._get_viewer().render()
            data = _get_image_hack(self._get_viewer())
            return data
        elif mode == 'human':
            self._get_viewer().loop_once()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
            # self.viewer.start()
            # self.viewer.set_model(self.model)
            # self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        idx = self.data.body_names.index(six.b(body_name))
        return self.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.data.body_names.index(six.b(body_name))
        return self.data.body_comvels[idx]

    def get_body_xmat(self, body_name):
        idx = self.data.body_names.index(six.b(body_name))
        return self.data.data.xmat[idx].reshape((3, 3))

    def state_vector(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ])
