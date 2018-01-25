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
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        achieved_goal = self._get_achieved_goal().flatten()
        observation = np.concatenate([robot_qpos, robot_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'goal': self.goal.flatten().copy(),
        }

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
            self, 'reach.xml', n_substeps=20, initial_qpos=initial_qpos, relative_control=False,
            dist_threshold=0.02)
        utils.EzPickle.__init__(self)

    def initial_setup(self):
        super(ReachEnv, self).initial_setup()
        self.palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:palm')].copy()

    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal)

    def _reset_goal(self):
        thumb_name = 'robot0:S_thtip'
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]
        finger_name = np.random.choice(finger_names)

        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
        assert thumb_idx != finger_idx

        # Pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
        meeting_pos += np.random.normal(scale=0.005, size=meeting_pos.shape)

        # Move the meeting point 30% towards the target finger.
        
        # Slightly move meeting goal towards the respective finger to avoid that they
        # overlap.
        goal = self.initial_goal.copy()
        for idx in [thumb_idx, finger_idx]:
            offset_direction = (meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.005 * offset_direction

        if np.random.random() < 0.1:
            # With some probability, ask all fingers to move back to the origin.
            # This avoids that the thumb constantly stays near the goal position already.
            goal = self.initial_goal.copy()
        self.goal = goal

    def subtract_goals(self, goal_a, goal_b):
        # In this case, our goal subtraction is quite simple since it does not
        # contain any rotations but only positions.
        assert goal_a.shape == goal_b.shape
        return goal_a - goal_b

    def _render_callback(self):
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


class BlockEnv(HandEnv, utils.EzPickle):
    def __init__(self, target_pos, target_rot, pos_mul):
        self.target_pos = target_pos
        self.target_rot = target_rot
        self.pos_mul = pos_mul

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
            self, 'block.xml', n_substeps=20, initial_qpos=initial_qpos, relative_control=False,
            dist_threshold=0.02)
        utils.EzPickle.__init__(self)

    # def _randomize_initial_position(self):
    #     # randomize rot
    #     initial_rot = self.sim.data.qpos[self._cube_angle_idxs]
    #     uniform_rot = np.random.uniform(0.0, 2 * np.pi, size=(3,))
    #     if self.target_rot == 'z':
    #         rot = subtract_euler(np.concatenate([np.zeros(2), [uniform_rot[2]]]), initial_rot)
    #     elif self.target_rot in ['xyz', 'parallel', 'ignore']:
    #         rot = uniform_rot
    #     else:  # fixed
    #         rot = initial_rot
    #     self.sim.data.qpos[self._cube_angle_idxs] = rot

    #     # randomize pos
    #     if self.target_pos != 'fixed':
    #         self.sim.data.qpos[self._cube_pos_idxs] += np.random.randn(3) * self.cube_position_wiggle_std

    #     # do not randomize face rotation for now
    #     if self.cube_type in ['face', 'full']:
    #         self.sim.data.qpos[self._cube_face_idxs] = 0

    #     action = self.random_state.uniform(-1.0, 1.0, 20)
    #     for _ in range(self.n_random_initial_steps):
    #         dactyl_simple_set_action(self.sim, action)
    #         self.sim.step()

    def _get_block_qpos(self, qpos):
        joint_names = ['block_tx', 'block_ty', 'block_tz', 'block_rx', 'block_ry', 'block_rz']
        block_qpos = []
        for name in joint_names:
            addr = self.sim.model.get_joint_qpos_addr('block:{}'.format(name))
            block_qpos.append(qpos[addr])
        block_qpos = np.array(block_qpos)
        assert block_qpos.shape == (6,)
        return block_qpos

    def _reset_goal(self):
        # Start with the initial state of the block.
        goal = self._get_block_qpos(self.initial_state.qpos).copy()

        # Select a goal for the block position.
        target_pos = np.zeros(3)
        if self.target_pos == 'random':
            target_pos += np.random.uniform(self.target_range[:, 0], self.target_range[:, 1])
        elif self.target_pos == 'ignore' or self.pos_mul == 0.:
            target_pos[:] = 0.
        elif self.target_pos == 'fixed':
            target_pos = self._get_block_qpos(self.sim.data.qpos)[:3]
        else:
            raise error.Error('Unknown target_pos option "{}".'.format(self.target_pos))
        assert target_pos.shape == (3,)

        # Select a goal for the block rotation.
        target_rot = np.zeros(3)
        if self.target_rot == 'z':
            target_rot[-1] = np.random.uniform(-np.pi, np.pi)
        elif self.target_rot == 'parallel':
            raise NotImplementedError()
            # target_rot = np.random.uniform(-math.pi, math.pi, size=3)
            # target_rot[:2] = 0
            # parallel_rot = self.parallel_rotations[self.random_state.randint(len(self.parallel_rotations))]
            # target_rot = mat2euler(np.matmul(euler2mat(target_rot), euler2mat(parallel_rot)))
        elif self.target_rot == 'xyz':
            target_rot = np.random.uniform(-np.pi, np.pi, size=3)
        elif self.target_rot == 'ignore':
            target_rot[:] = 0.
        elif self.target_rot == 'fixed':
            target_rot = self._get_block_qpos(self.sim.data.qpos)[3:]
        else:
            raise error.Error('Unknown target_rot option "{}".'.format(self.target_rot))
        assert target_rot.shape == (3,)

        # TODO: do we need this?
        # if self.round_target_rot:
        #     target_rot = round_to_straight_angles(target_rot)
        goal = np.concatenate([target_pos, rotations.normalize_angles(target_rot)])
        self.goal = goal.copy()

    def _get_achieved_goal(self):
        # Block position and rotation.
        block_qpos = self._get_block_qpos(self.sim.data.qpos)
        assert block_qpos.shape == (6,)
        block_qpos[3:] = rotations.normalize_angles(block_qpos[3:])
        return block_qpos.copy()

    def _render_callback(self):
        joint_names_pos = ['target:block_tx', 'target:block_ty', 'target:block_tz']
        joint_names_rot = ['target:block_rx', 'target:block_ry', 'target:block_rz']

        # Assign current state to target block but offset a bit so that the actual block
        # is not obscured.
        goal = self.goal.copy()
        goal[0] += 0.15
        for name, value in zip(joint_names_pos[:] + joint_names_rot[:], goal):
            self.sim.data.set_joint_qpos(name, value)
            self.sim.data.set_joint_qvel(name, 0.)

    def subtract_goals(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 6
            
        diff = goal_a - goal_b
        diff[..., :3] *= float(self.target_pos != 'ignore') * self.pos_mul  # position
        if self.target_rot == 'ignore':
            diff[..., 3:] = 0.
        else:
            # Only do this if we have to since this operation can be expensive
            diff[..., 3:] = rotations.subtract_euler(goal_a[..., 3:], goal_b[..., 3:])  # orientation
        return diff


class BlockRotateXYZEnv(BlockEnv):
    def __init__(self):
        super(BlockRotateXYZEnv, self).__init__(
            target_pos='ignore', target_rot='xyz', pos_mul=0.)


class BlockRotateZEnv(BlockEnv):
    def __init__(self):
        super(BlockRotateZEnv, self).__init__(
            target_pos='ignore', target_rot='z', pos_mul=0.)
