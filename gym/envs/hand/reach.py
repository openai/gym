import os
import copy

from gym import error, spaces
from gym.utils import seeding
from mujoco_py import const
import numpy as np
from os import path
import gym
import six
from gym import utils

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

        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.initial_goal = self._get_achieved_goal().copy()
        self.palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:palm')].copy()

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
        observation = np.concatenate([self.sim.data.qpos, self.sim.data.qvel])
        return {
            'observation': observation.copy(),
            'achieved_goal': self._get_achieved_goal().flatten().copy(),
            'goal': self.goal.flatten().copy(),
        }

    # Goal-based API

    def _reset_goal(self):
        thumb_name = 'robot0:S_thtip'
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]
        finger_name = np.random.choice(finger_names)

        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
        assert thumb_idx != finger_idx

        # Pick a meeting point atop the hand.
        meeting_pos = self.palm_xpos + np.array([-0.01, -0.03, 0.08])
        meeting_pos += np.random.normal(scale=0.005, size=meeting_pos.shape)

        # Move the meeting point towards the target finger.
        goal = self.initial_goal.copy()
        offset_direction = (goal[finger_idx] - meeting_pos)
        offset_direction /= np.linalg.norm(offset_direction)
        meeting_pos = meeting_pos + 0.05 * offset_direction

        # Slightly move meeting goal towards the respective finger to avoid that they
        # overlap.
        for idx in [thumb_idx, finger_idx]:
            offset_direction = (meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.005 * offset_direction

        if np.random.random() < 0.1:
            # With some probability, ask all fingers to move back to the origin.
            # This avoids that the thumb constantly stays near the goal position already.
            goal = self.initial_goal.copy()
        self.goal = goal

    def _compute_goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(self.subtract_goals(goal_a, goal_b), axis=-1)

    def subtract_goals(self, goal_a, goal_b):
        # In this case, our goal subtraction is quite simple since it does not
        # contain any rotations but only positions.
        assert goal_a.shape == goal_b.shape
        return goal_a - goal_b

    def is_success(self, achieved_goal, goal):
        d = self._compute_goal_distance(achieved_goal, goal)
        return (d < self.dist_threshold).astype(np.float32)

    def compute_reward(self, obs, action, next_obs, goal):
        # Compute distance between goal and the achieved goal.
        d = self._compute_goal_distance(next_obs['achieved_goal'], goal)
        return -(d > self.dist_threshold).astype(np.float32)

    # -----------------------------

    def _reset(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        self._reset_goal()
        obs = self._get_obs()
        if self.viewer is not None:
            self.viewer_setup()
        return obs

    @property
    def dt(self):
        return self.model.opt.timestep * self.n_substeps

    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal)


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer()
                self.viewer = None
            return

        # Visualize targets.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        for finger_idx in range(5):
            site_name = 'target{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = self.goal[finger_idx] - sites_offset[site_id]

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal()
        for finger_idx in range(5):
            site_name = 'finger{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]

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


class ReachEnv(HandEnv, utils.EzPickle):
    def __init__(self):
        initial_qpos = {
            'robot0:WRJ1': -0.16514339750464327,
            'robot0:WRJ0': -0.31973286565062153,
            'robot0:FFJ3': 0.14340512546557435,
            'robot0:FFJ2': 0.32028208333591573,
            'robot0:FFJ1': 0.7126053607727917,
            'robot0:FFJ0': 0.6705281001412586,
            'robot0:MFJ3': 0.000246444303701037,
            'robot0:MFJ2': 0.3152655251085491,
            'robot0:MFJ1': 0.7659800313729842,
            'robot0:MFJ0': 0.7323156897425923,
            'robot0:RFJ3': 0.00038520700007378114,
            'robot0:RFJ2': 0.36743546201985233,
            'robot0:RFJ1': 0.7119514095008576,
            'robot0:RFJ0': 0.6699446327514138,
            'robot0:LFJ4': 0.0525442258033891,
            'robot0:LFJ3': -0.13615534724474673,
            'robot0:LFJ2': 0.39872030433433003,
            'robot0:LFJ1': 0.7415570009679252,
            'robot0:LFJ0': 0.704096378652974,
            'robot0:THJ4': 0.003673823825070126,
            'robot0:THJ3': 0.5506291436028695,
            'robot0:THJ2': -0.014515151997119306,
            'robot0:THJ1': -0.0015229223564485414,
            'robot0:THJ0': -0.7894883021600622,
        }
        HandEnv.__init__(
            self, 'reach.xml', n_substeps=20, initial_qpos=initial_qpos, relative_control=True,
            dist_threshold=0.05)
        utils.EzPickle.__init__(self)
