from collections import OrderedDict
from os import path
from typing import Optional

import numpy as np

import gym
from gym import error, logger, spaces

DEFAULT_SIZE = 480


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

    def __init__(self, model_path, frame_skip, mujoco_bindings="mujoco"):

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")

        if mujoco_bindings == "mujoco_py":
            logger.warn(
                "This version of the mujoco environments depends "
                "on the mujoco-py bindings, which are no longer maintained "
                "and may stop working. Please upgrade to the v4 versions of "
                "the environments (which depend on the mujoco python bindings instead), unless "
                "you are trying to precisely replicate previous works)."
            )
            try:
                import mujoco_py

                self._mujoco_bindings = mujoco_py
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
                        e
                    )
                )

            self.model = self._mujoco_bindings.load_model_from_path(fullpath)
            self.sim = self._mujoco_bindings.MjSim(self.model)
            self.data = self.sim.data

        elif mujoco_bindings == "mujoco":
            try:
                import mujoco

                self._mujoco_bindings = mujoco

            except ImportError as e:
                raise error.DependencyNotInstalled(
                    f"{e}. (HINT: you need to install mujoco)"
                )
            self.model = self._mujoco_bindings.MjModel.from_xml_path(fullpath)
            self.data = self._mujoco_bindings.MjData(self.model)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._viewers = {}

        self.frame_skip = frame_skip

        self.viewer = None

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self._set_action_space()

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
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """

    # -----------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        if self._mujoco_bindings.__name__ == "mujoco_py":
            self.sim.reset()
        else:
            self._mujoco_bindings.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        if not return_info:
            return ob
        else:
            return ob, {}

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        if self._mujoco_bindings.__name__ == "mujoco_py":
            state = self.sim.get_state()
            state = self._mujoco_bindings.MjSimState(
                state.time, qpos, qvel, state.act, state.udd_state
            )
            self.sim.set_state(state)
            self.sim.forward()
        else:
            self.data.qpos[:] = np.copy(qpos)
            self.data.qvel[:] = np.copy(qvel)
            if self.model.na == 0:
                self.data.act[:] = None
            self._mujoco_bindings.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        if self._mujoco_bindings.__name__ == "mujoco_py":
            self.sim.data.ctrl[:] = ctrl
        else:
            self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            if self._mujoco_bindings.__name__ == "mujoco_py":
                self.sim.step()
            else:
                self._mujoco_bindings.mj_step(self.model, self.data)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        if self._mujoco_bindings.__name__ != "mujoco_py":
            self._mujoco_bindings.mj_rnePostConstraint(self.model, self.data)

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "rgb_array" or mode == "depth_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                if self._mujoco_bindings.__name__ == "mujoco_py":
                    if camera_name in self.model._camera_name2id:
                        camera_id = self.model.camera_name2id(camera_name)
                else:
                    camera_id = self._mujoco_bindings.mj_name2id(
                        self.model,
                        self._mujoco_bindings.mjtObj.mjOBJ_CAMERA,
                        camera_name,
                    )

                self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == "rgb_array":
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            if self._mujoco_bindings.__name__ == "mujoco":
                self.viewer.close()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode, width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                if self._mujoco_bindings.__name__ == "mujoco_py":
                    self.viewer = self._mujoco_bindings.MjViewer(self.sim)
                else:
                    from gym.envs.mujoco.mujoco_rendering import Viewer

                    self.viewer = Viewer(self.model, self.data)
            elif mode == "rgb_array" or mode == "depth_array":
                if self._mujoco_bindings.__name__ == "mujoco_py":
                    self.viewer = self._mujoco_bindings.MjRenderContextOffscreen(
                        self.sim, -1
                    )
                else:
                    from gym.envs.mujoco.mujoco_rendering import RenderContextOffscreen

                    self.viewer = RenderContextOffscreen(
                        width, height, self.model, self.data
                    )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        if self._mujoco_bindings.__name__ == "mujoco_py":
            return self.data.get_body_xpos(body_name)
        else:
            return self.data.body(body_name).xpos

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
