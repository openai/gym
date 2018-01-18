import os
import copy

from gym import error, spaces
from gym.utils import seeding
from mujoco_py import const
import numpy as np
from os import path
import gym
import six

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


class FetchEnv(gym.Env):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_box, target_in_the_air, target_x_shift, obj_range, target_range, dist_threshold):
        # TODO: n_substeps
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        
        self.n_substeps = n_substeps
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_box = has_box
        self.target_in_the_air = target_in_the_air
        self.target_x_shift = target_x_shift
        self.obj_range = obj_range
        self.target_range = target_range
        self.dist_threshold = dist_threshold

        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.data = self.sim.data
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.initial_setup()
        self.init_state = copy.deepcopy(self.sim.get_state())

        self.action_space = spaces.Box(-np.inf, np.inf, 4)

        self.reset()
        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict([(k, spaces.Box(-np.inf, np.inf, v.size)) for k, v in obs.items()]))
        print(self.observation_space)
        
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

        # Places mocap where related bodies are.
        if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
            for i in range(self.sim.model.eq_data.shape[0]):
                if self.sim.model.eq_type[i] == const.EQ_WELD:
                    self.sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        
        # Move end effector into position.
        self.sim.forward()
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(100):
            self.sim.step()

        # Extract information for sampling goals.
        self.gripper = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_box:
            self.height_offset = self.sim.data.get_site_xpos('geom0')[2]

    @property
    def initial_qpos(self):
        raise NotImplementedError()

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

    def _step(self, action):
        self._set_action(action)
        obs = self._get_obs()
        self.sim.step()

        reward = self.compute_reward(obs)
        done = False
        return obs, reward, done, {}

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
            'obs': obs.copy(),
            'goal': self.goal.copy(),
            'achieved_goal': achieved_goal.copy(),
        }

    def reset_goal(self):
        if not self.has_box:
            goal = self.gripper[:3] + np.random.uniform(-0.15, 0.15, size=3)
        else:
            box_xpos = self.gripper[:2]
            while np.linalg.norm(box_xpos - self.gripper[:2]) < 0.1:
                box_xpos = self.gripper[:2] + np.random.uniform(-self.obj_range, self.obj_range, size=2)
            goal = self.gripper[:3] + np.random.uniform(-self.target_range, self.target_range, size=3)
            goal[0] += self.target_x_shift
            goal[2] = self.height_offset
            if self.target_in_the_air and np.random.uniform() < 0.5:
                goal[2] += np.random.uniform(0, 0.45)
            qpos = self.init_state.qpos
            qpos[-6:-4] = box_xpos
            qpos[-3:] = 0.  # no rotation
        self.goal = goal

        # Set site position for visualization.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]

    def subtract_goals(self, a, b):
        return a - b

    def goal_distance(self, obs, goal):
        current_goal = obs['achieved_goal']
        assert current_goal.shape == goal.shape
        return np.linalg.norm(self.subtract_goals(current_goal, goal), axis=-1)

    def compute_reward(self, obs):
        d = self.goal_distance(obs, self.goal)
        return -(d > self.dist_threshold).astype(np.float32)

    # -----------------------------

    def _reset(self):
        self.sim.reset()
        self.sim.set_state(self.init_state)
        self.sim.forward()
        self.reset_goal()
        obs = self._get_obs()
        if self.viewer is not None:
            self.viewer_setup()
        return obs

    @property
    def dt(self):
        return self.model.opt.timestep * self.n_substeps

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer()
                self.viewer = None
            return

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
