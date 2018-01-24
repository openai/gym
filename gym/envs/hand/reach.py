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


def ctrl_set_action(sim, action):
    """
    For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mocap_set_action(sim, action):
    """
    The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    contraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        # TODO: this split should probably happen in simple_set_action
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap2body_xpos(sim):
    """
    Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """
    if sim.model.eq_type is None or \
       sim.model.eq_obj1id is None or \
       sim.model.eq_obj2id is None:
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


def robot_get_obs(sim):
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (np.array([sim.data.get_joint_qpos(name) for name in names]),
                np.array([sim.data.get_joint_qvel(name) for name in names]))
    return np.zeros(0), np.zeros(0)


def mat2euler(mat):
    """
    Converts a rotation matrix (or a batch thereof) to euler angles.
    """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > np.finfo(np.float64).eps * 4.
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler


def set_action(sim, action):
    ctrl_set_action(sim, action)
    mocap_set_action(sim, action)


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

    def __init__(self, model_path, n_substeps, initial_qpos):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        
        self.n_substeps = n_substeps
        self.initial_qpos = initial_qpos

        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.initial_setup()
        
        self.action_space = spaces.Box(-np.inf, np.inf, 4)

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
        init_qpos = self.initial_qpos
        for name, value in init_qpos.items():
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
        #self._set_action(action)
        self.sim.step()
        next_obs = self._get_obs()

        reward = self.compute_reward(obs, action, next_obs, self.goal)
        done = False
        return next_obs, reward, done, {}

    def _set_action(self, action):
        assert action.shape == (4,)
        gripper_ctrl = action[3]
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([action[:3], [1., 0., 1., 0.]])
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([action, gripper_ctrl])
        set_action(self.sim, action)

    def _get_obs(self):
        return {
            'observation': np.zeros(3),
            'achieved_goal': np.zeros(3),
            'goal': np.zeros(3),
        }
        
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        if self.has_box:
            box_pos = self.sim.data.get_site_xpos('geom0')
            # rotations
            box_rot = mat2euler(self.sim.data.get_site_xmat('geom0'))
            # velocities
            box_velp = self.sim.data.get_site_xvelp('geom0') * dt
            box_velr = self.sim.data.get_site_xvelr('geom0') * dt
            # gripper state
            box_rel_pos = box_pos - grip_pos
            box_velp -= grip_velp
        else:
            box_pos = box_rot = box_velp = box_velr = box_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        obs = np.concatenate([grip_pos, box_rel_pos.flatten(), gripper_state,
                                 box_rot.flatten(), box_velp.flatten(), box_velr.flatten(),
                                 grip_velp, gripper_vel])

        if not self.has_box:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(box_pos.copy())

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'goal': self.goal.copy(),
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
        pass

    def subtract_goals(self, goal_a, goal_b):
        pass

    def is_success(self, achieved_goal, goal):
        return 0.

    def compute_reward(self, obs, action, next_obs, goal):
        return 0.

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

        self.sim.forward()  # TODO: remove eventually

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
        }
        HandEnv.__init__(
            self, 'reach.xml', n_substeps=20, initial_qpos=initial_qpos)
        utils.EzPickle.__init__(self)
