from functools import partial
from os import path
from typing import Optional, Union

import numpy as np

import gym
from gym import error, logger, spaces
from gym.spaces import Space
from gym.utils.renderer import Renderer

MUJOCO_PY_NOT_INSTALLED = False
MUJOCO_NOT_INSTALLED = False

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
    MUJOCO_PY_NOT_INSTALLED = True

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
    MUJOCO_NOT_INSTALLED = True


DEFAULT_SIZE = 480


class BaseMujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        if model_path.startswith("/"):
            self.fullpath = model_path
        else:
            self.fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        self._initialize_simulation()

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._viewers = {}

        self.frame_skip = frame_skip

        self.viewer = None

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ]
        assert (
            int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.observation_space = observation_space
        self._set_action_space()

        self.render_mode = render_mode
        render_frame = partial(
            self._render,
            width=width,
            height=height,
            camera_name=camera_name,
            camera_id=camera_id,
        )
        self.renderer = Renderer(self.render_mode, render_frame)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

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
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """

    def _initialize_simulation(self):
        """
        Initialize MuJoCo simulation data structures mjModel and mjData.
        """
        raise NotImplementedError

    def _reset_simulation(self):
        """
        Reset MuJoCo simulation data structures, mjModel and mjData.
        """
        raise NotImplementedError

    def _step_mujoco_simulation(self, ctrl, n_frames):
        """
        Step over the MuJoCo simulation.
        """
        raise NotImplementedError

    def _render(
        self,
        mode: str = "human",
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        raise NotImplementedError

    # -----------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self._reset_simulation()

        ob = self.reset_model()
        self.renderer.reset()
        self.renderer.render_step()
        if not return_info:
            return ob
        else:
            return ob, {}

    def set_state(self, qpos, qvel):
        """
        Set the joints position qpos and velocity qvel of the model. Override this method depending on the MuJoCo bindings used.
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        self._step_mujoco_simulation(ctrl, n_frames)

    def render(
        self,
        mode: str = "human",
        width: Optional[int] = None,
        height: Optional[int] = None,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        if self.render_mode is not None:
            assert (
                width is None
                and height is None
                and camera_id is None
                and camera_name is None
            ), "Unexpected argument for render. Specify render arguments at environment initialization."
            return self.renderer.get_renders()
        else:
            width = width if width is not None else DEFAULT_SIZE
            height = height if height is not None else DEFAULT_SIZE
            return self._render(
                mode=mode,
                width=width,
                height=height,
                camera_id=camera_id,
                camera_name=camera_name,
            )

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def get_body_com(self, body_name):
        """Return the cartesian position of a body frame"""
        raise NotImplementedError

    def state_vector(self):
        """Return the position and velocity joint states of the model"""
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])


class MuJocoPyEnv(BaseMujocoEnv):
    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        if MUJOCO_PY_NOT_INSTALLED:
            raise error.DependencyNotInstalled(
                f"{MUJOCO_PY_IMPORT_ERROR}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)"
            )

        logger.warn(
            "This version of the mujoco environments depends "
            "on the mujoco-py bindings, which are no longer maintained "
            "and may stop working. Please upgrade to the v4 versions of "
            "the environments (which depend on the mujoco python bindings instead), unless "
            "you are trying to precisely replicate previous works)."
        )

        super().__init__(
            model_path,
            frame_skip,
            observation_space,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
        )

    def _initialize_simulation(self):
        self.model = mujoco_py.load_model_from_path(self.fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data

    def _reset_simulation(self):
        self.sim.reset()

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        state = self.sim.get_state()
        state = mujoco_py.MjSimState(state.time, qpos, qvel, state.act, state.udd_state)
        self.sim.set_state(state)
        self.sim.forward()

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            self.sim.step()

    def _render(
        self,
        mode: str = "human",
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        assert mode in self.metadata["render_modes"]
        if mode in {
            "rgb_array",
            "single_rgb_array",
            "depth_array",
            "single_depth_array",
        }:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                if camera_name in self.model._camera_name2id:
                    camera_id = self.model.camera_name2id(camera_name)

                self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode in {"rgb_array", "single_rgb_array"}:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode in {"depth_array", "single_depth_array"}:
            self._get_viewer(mode).render(width, height)
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(
        self, mode
    ) -> Union["mujoco_py.MjViewer", "mujoco_py.MjRenderContextOffscreen"]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)

            elif mode in {
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            }:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
            else:
                raise AttributeError(
                    f"Unknown mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer

        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)


class MujocoEnv(BaseMujocoEnv):
    """Superclass for MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        if MUJOCO_NOT_INSTALLED:
            raise error.DependencyNotInstalled(
                f"{MUJOCO_IMPORT_ERROR}. (HINT: you need to install mujoco)"
            )
        super().__init__(
            model_path,
            frame_skip,
            observation_space,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
        )

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _render(
        self,
        mode: str = "human",
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        assert mode in self.metadata["render_modes"]

        if mode in {
            "rgb_array",
            "single_rgb_array",
            "depth_array",
            "single_depth_array",
        }:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

                self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode in {"rgb_array", "single_rgb_array"}:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode in {"depth_array", "single_depth_array"}:
            self._get_viewer(mode).render(width, height)
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        super().close()

    def _get_viewer(
        self, mode, width=DEFAULT_SIZE, height=DEFAULT_SIZE
    ) -> Union["gym.envs.mujoco.Viewer", "gym.envs.mujoco.RenderContextOffscreen"]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                from gym.envs.mujoco import Viewer

                self.viewer = Viewer(self.model, self.data)
            elif mode in {
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            }:
                from gym.envs.mujoco import RenderContextOffscreen

                self.viewer = RenderContextOffscreen(
                    width, height, self.model, self.data
                )
            else:
                raise AttributeError(
                    f"Unexpected mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos
