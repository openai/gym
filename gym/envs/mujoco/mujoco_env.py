import os
from collections import OrderedDict
from os import path
from typing import Optional

import numpy as np

import gym
from gym import error, spaces
from gym.utils.renderer import Renderer

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )

DEFAULT_SIZE = 500


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        render_mode: Optional[str] = None,
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.width = width
        self.height = height
        self.camera_id = camera_id
        self.camera_name = camera_name

        if self.render_mode == "rgb_array" or self.render_mode == "depth_array":
            if self.camera_id is not None and self.camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            if self.camera_name is None and self.camera_id is None:
                self.camera_name = "track"

            if (
                self.camera_id is None
                and self.camera_name in self.model._camera_name2id
            ):
                self.camera_id = self.model.camera_name2id(self.camera_name)

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

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
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.sim.reset()
        ob = self.reset_model()
        self.renderer.reset()
        self.renderer.render_step()
        if not return_info:
            return ob
        else:
            return ob, {}

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode):
        assert mode in self.metadata["render_modes"]

        if mode in ["rgb_array", "single_rgb_array"]:
            self._get_viewer(mode).render(
                self.width, self.height, camera_id=self.camera_id
            )
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(
                self.width, self.height, depth=False
            )
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode in ["depth_array", "single_depth_array"]:
            self._get_viewer(mode).render(
                self.width, self.height, camera_id=self.camera_id
            )
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(
                self.width, self.height, depth=True
            )[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode in [
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            ]:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
