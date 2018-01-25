import os
import copy

from gym import error, spaces
from gym.utils import seeding
from mujoco_py import const
import numpy as np
from os import path
import gym
import six
from gym import utils, error

from gym.envs.hand import rotations

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def robot_get_obs(sim):
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (np.array([sim.data.get_joint_qpos(name) for name in names]),
                np.array([sim.data.get_joint_qvel(name) for name in names]))
    return np.zeros(0), np.zeros(0)


def set_action(sim, action, relative=False):
    ctrlrange = sim.model.actuator_ctrlrange
    actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
    if relative:
        actuation_center = np.zeros_like(action)
        for i in range(sim.data.ctrl.shape[0]):
            actuation_center[i] = sim.data.get_joint_qpos(sim.model.actuator_names[i].replace(':A_', ':'))
        for joint_name in ['FF', 'MF', 'RF', 'LF']:
            act_idx = sim.model.actuator_name2id('robot0:A_{}J1'.format(joint_name))
            actuation_center[act_idx] += sim.data.get_joint_qpos("robot0:{}J0".format(joint_name))
    else:
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.

    sim.data.ctrl[:] = actuation_center + action * actuation_range
    sim.data.ctrl[:] = np.clip(sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])


FINGERTIP_SITE_NAMES = [
    'robot0:S_fftip',
    'robot0:S_mftip',
    'robot0:S_rftip',
    'robot0:S_lftip',
    'robot0:S_thtip',
]


class HandEnv(gym.GoalEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(self, model_path, n_substeps, initial_qpos, relative_control, dist_threshold):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        
        self.n_substeps = n_substeps
        self.initial_qpos = initial_qpos
        self.relative_control = relative_control
        self.dist_threshold = dist_threshold

        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.initial_setup()
        self._get_initial_state()
        
        self.action_space = spaces.Box(-np.inf, np.inf, 20)

        self._reset_goal()
        obs = self._get_obs()
        self.observation_space = spaces.Goal(
            goal_space=spaces.Box(-np.inf, np.inf, obs['achieved_goal'].size),
            observation_space=spaces.Box(-np.inf, np.inf, obs['observation'].size),
        )

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def initial_setup(self):
        """
        Custom setup (e.g. setting initial qpos, moving into desired start position, etc.)
        can be done here
        """
        for name, value in self.initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _get_initial_state(self):
        self.initial_state = copy.deepcopy(self.sim.get_state())

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def _step(self, action):
        obs = self._get_obs()
        self._set_action(action)
        self.sim.step()
        next_obs = self._get_obs()

        reward = self.compute_reward(obs, action, next_obs, self.goal.flatten())
        done = False
        return next_obs, reward, done, {}

    def _set_action(self, action):
        assert action.shape == (20,)
        set_action(self.sim, action, relative=self.relative_control)

    def _get_obs(self):
        raise NotImplementedError()

    # Goal-based API

    def _reset_goal(self):
        raise NotImplementedError()

    def _compute_goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(self.subtract_goals(goal_a, goal_b), axis=-1)

    def is_success(self, achieved_goal, goal):
        d = self._compute_goal_distance(achieved_goal, goal)
        return (d < self.dist_threshold).astype(np.float32)

    def compute_reward(self, obs, action, next_obs, goal):
        # Compute distance between goal and the achieved goal.
        d = self._compute_goal_distance(next_obs['achieved_goal'], goal)
        return -(d > self.dist_threshold).astype(np.float32)

    # -----------------------------

    def _reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_simulation()
        
        self._reset_goal()
        obs = self._get_obs()
        if self.viewer is not None:
            self.viewer_setup()
        return obs

    def _reset_simulation(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    @property
    def dt(self):
        return self.model.opt.timestep * self.n_substeps

    def _get_achieved_goal(self):
        raise NotImplementedError()

    def _render_callback(self):
        pass

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer()
                self.viewer = None
            return

        self._render_callback()

        if mode == 'rgb_array':
            self._get_viewer().render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer
