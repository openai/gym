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

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip, resolution=(240, 240)):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.resolution = resolution
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
            return  # deprecated
        if mode == 'rgb_array':
            return self.sim.render(self.resolution[0], self.resolution[1])
        elif mode == 'human':
            self.sim.render(mode='window')

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
