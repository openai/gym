import numpy as np

from gym import utils, error
from gym.envs.robotics import rotations, hand_env
from gym.envs.robotics.utils import robot_get_obs

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def get_block_qpos(sim, qpos):
    joint_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    block_qpos = []
    for name in joint_names:
        addr = sim.model.get_joint_qpos_addr('object:{}'.format(name))
        block_qpos.append(qpos[addr])
    block_qpos = np.array(block_qpos)
    assert block_qpos.shape == (6,)
    return block_qpos


def set_block_qpos(sim, qpos):
    assert qpos.shape == (6,)
    joint_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    for name, value in zip(joint_names, qpos):
        addr = sim.model.get_joint_qpos_addr('object:{}'.format(name))
        sim.data.qpos[addr] = value


def get_block_qvel(sim, qvel):
    joint_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    block_qvel = []
    for name in joint_names:
        addr = sim.model.get_joint_qvel_addr('object:{}'.format(name))
        block_qvel.append(qvel[addr])
    block_qvel = np.array(block_qvel)
    assert block_qvel.shape == (6,)
    return block_qvel


class ManipulateEnv(hand_env.HandEnv, utils.EzPickle):
    def __init__(
        self, model_path, target_position, target_rotation, position_weight,
        target_position_range, reward_type, initial_qpos={},
        randomize_initial_position=True, randomize_initial_rotation=True,
        distance_threshold=0.4, n_substeps=20, relative_control=False,
        rotation_ignore_mask=np.zeros(3, dtype='bool'),
    ):
        """Initializes a new Hand manipulation environment.

        Args:
            model_path (string): path to the environments XML file
            target_position (string): the type of target position:
                - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                - fixed: target position is set to the initial position of the object
                - random: target position is fully randomized according to target_position_range
            target_rotation (string): the type of target rotation:
                - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                - fixed: target rotation is set to the initial rotation of the object
                - xyz: fully randomized target rotation around the X, Y and Z axis
                - xy: fully randomized target rotation around the X and Y axis
                - z: fully randomized target rotation around the Z axis
                - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y
            position_weight (float): the weight of the position offset when computing goal differences
            target_position_range (np.array of shape (3, 2)): range of the target_position randomization
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            distance_threshold (float): the threshold after which a goal is considered achieved
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state
            rotation_ignore_mask (np.array of shape (3,)): ignores the rotation axis in distance computations if corresponding index evaluates to True
        """
        self.target_position = target_position
        self.target_rotation = target_rotation
        self.position_weight = position_weight
        self.target_position_range = target_position_range
        self.parallel_rotations = rotations.get_parallel_rotations()
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.rotation_ignore_mask = rotation_ignore_mask

        assert self.target_position in ['ignore', 'fixed', 'random']
        assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'xy', 'z', 'parallel']

        hand_env.HandEnv.__init__(
            self, model_path, n_substeps=n_substeps, initial_qpos=initial_qpos,
            relative_control=relative_control)
        utils.EzPickle.__init__(self)

    def _get_achieved_goal(self):
        # Block position and rotation.
        block_qpos = get_block_qpos(self.sim, self.sim.data.qpos)
        assert block_qpos.shape == (6,)
        block_qpos[3:] = rotations.normalize_angles(block_qpos[3:])
        return block_qpos.copy()

    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 6

        d_pos = 0.
        d_rot = 0.
        if self.target_position != 'ignore':
            delta_pos = (goal_a[..., :3] - goal_b[..., :3]) * self.position_weight
            d_pos = np.linalg.norm(delta_pos, axis=-1)
        if self.target_rotation != 'ignore':
            assert self.rotation_ignore_mask.shape == (3,)
            euler_a, euler_b = goal_a[..., 3:].copy(), goal_b[..., 3:].copy()
            for idx, value in enumerate(self.rotation_ignore_mask):
                if value:
                    euler_a[..., idx] = 0.
                    euler_b[..., idx] = 0.
            delta_rot = rotations.subtract_euler(euler_a, euler_b)
            d_rot = np.linalg.norm(delta_rot, axis=-1)
        d = d_pos + d_rot
        return d

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        d = self._goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, desired_goal):
        d = self._goal_distance(achieved_goal, desired_goal)
        is_success = (d < self.distance_threshold).astype(np.float32)
        return is_success

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = get_block_qpos(self.sim, self.initial_state.qpos).copy()

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            uniform_rot = self.np_random.uniform(0.0, 2 * np.pi, size=(3,))
            if self.target_rotation == 'z':
                initial_qpos[3:] = rotations.subtract_euler(np.concatenate([np.zeros(2), [uniform_rot[2]]]), initial_qpos[3:])
            elif self.target_rotation == 'xy':
                initial_qpos[3:] = rotations.subtract_euler(np.concatenate([uniform_rot[2], [0.]]), initial_qpos[3:])
            elif self.target_rotation in ['xyz', 'parallel', 'ignore']:
                initial_qpos[3:] = uniform_rot
            elif self.target_rotation == 'fixed':
                pass
            else:
                raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                initial_qpos[:3] += self.np_random.normal(size=3, scale=0.005)

        set_block_qpos(self.sim, initial_qpos)

        def is_on_palm():
            self.sim.forward()
            cube_middle_idx = self.sim.model.site_name2id('object:center')
            cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(20))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        return is_on_palm()

    def _sample_goal(self):
        # Select a goal for the block position.
        target_position = np.zeros(3)
        if self.target_position == 'random':
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
            assert offset.shape == (3,)
            target_position = get_block_qpos(self.sim, self.sim.data.qpos)[:3] + offset
        elif self.target_position == 'ignore' or self.position_weight == 0.:
            target_position[:] = 0.
        elif self.target_position == 'fixed':
            target_position = get_block_qpos(self.sim, self.sim.data.qpos)[:3]
        else:
            raise error.Error('Unknown target_position option "{}".'.format(self.target_position))
        assert target_position.shape == (3,)

        # Select a goal for the block rotation.
        target_rotation = np.zeros(3)
        if self.target_rotation == 'z':
            target_rotation[-1] = self.np_random.uniform(-np.pi, np.pi)
        elif self.target_rotation == 'parallel':
            target_rotation[-1] = self.np_random.uniform(-np.pi, np.pi)
            target_rotation[:2] = 0.
            parallel_rot = self.parallel_rotations[self.np_random.randint(len(self.parallel_rotations))]
            target_rotation = rotations.mat2euler(np.matmul(rotations.euler2mat(target_rotation), rotations.euler2mat(parallel_rot)))
        elif self.target_rotation == 'xyz':
            target_rotation = self.np_random.uniform(-np.pi, np.pi, size=3)
        elif self.target_rotation == 'xy':
            target_rotation[:2] = self.np_random.uniform(-np.pi, np.pi, size=2)
        elif self.target_rotation == 'ignore':
            target_rotation[:] = 0.
        elif self.target_rotation == 'fixed':
            target_rotation = get_block_qpos(self.sim, self.sim.data.qpos)[3:]
        else:
            raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))
        assert target_rotation.shape == (3,)

        goal = np.concatenate([target_position, rotations.normalize_angles(target_rotation)])
        return goal

    def _render_callback(self):
        joint_names_pos = ['target:tx', 'target:ty', 'target:tz']
        joint_names_rot = ['target:rx', 'target:ry', 'target:rz']

        # Assign current state to target block but offset a bit so that the actual block
        # is not obscured.
        goal = self.goal.copy()
        if self.target_position == 'ignore':
            # Move the block to the side since we do not care about it's position.
            goal[0] += 0.15
        for name, value in zip(joint_names_pos[:] + joint_names_rot[:], goal):
            self.sim.data.set_joint_qpos(name, value)
            self.sim.data.set_joint_qvel(name, 0.)

        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        block_qvel = get_block_qvel(self.sim, self.sim.data.qvel)
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the block position + rotation
        observation = np.concatenate([robot_qpos, robot_qvel, block_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }


class HandBlockEnv(ManipulateEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        super(HandBlockEnv, self).__init__(
            model_path='hand/manipulate_block.xml', target_position=target_position,
            target_rotation=target_rotation, position_weight=25.,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type)


class HandEggEnv(ManipulateEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        super(HandEggEnv, self).__init__(
            model_path='hand/manipulate_egg.xml', target_position=target_position,
            target_rotation=target_rotation, position_weight=25.,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type)


class HandPenEnv(ManipulateEnv):
    def __init__(self, target_position='random', target_rotation='xy', reward_type='sparse'):
        super(HandPenEnv, self).__init__(
            model_path='hand/manipulate_pen.xml', target_position=target_position,
            target_rotation=target_rotation, position_weight=25.,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            randomize_initial_rotation=False, reward_type=reward_type,
            rotation_ignore_mask=np.array([False, False, True]))
