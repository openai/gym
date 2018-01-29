import numpy as np

from gym import utils, error
from gym.envs.robotics import rotations, hand_env
from gym.envs.robotics.utils import robot_get_obs

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def goal_distance(goal_a, goal_b, position_weight, compute_rotation):
    assert goal_a.shape == goal_b.shape
    assert goal_a.shape[-1] == 6

    diff = goal_a - goal_b
    diff[..., :3] *= position_weight  # position
    if compute_rotation:
        diff[..., 3:] = rotations.subtract_euler(goal_a[..., 3:], goal_b[..., 3:])  # orientation
    else:
        diff[..., 3:] = 0.
    return np.linalg.norm(diff, axis=-1)


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
        target_position_range, initial_qpos={}, randomize_initial_position=True,
        randomize_initial_rotation=True, distance_threshold=0.4, n_substeps=20,
        relative_control=False
    ):
        self.target_position = target_position
        self.target_rotation = target_rotation
        self.position_weight = position_weight
        self.target_position_range = target_position_range
        self.parallel_rotations = rotations.get_parallel_rotations()
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold

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

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(
            achieved_goal, goal, position_weight=self.position_weight,
            compute_rotation=self.target_rotation != 'ignore')
        return -(d > self.distance_threshold).astype(np.float32)

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(
            achieved_goal, desired_goal, position_weight=self.position_weight,
            compute_rotation=self.target_rotation != 'ignore')
        return (d < self.distance_threshold).astype(np.float32)

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

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        block_qvel = get_block_qvel(self.sim, self.sim.data.qvel)
        achieved_goal = self._get_achieved_goal().flatten()  # this contains the block position + rotation
        observation = np.concatenate([robot_qpos, robot_qvel, block_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'goal': self.goal.flatten().copy(),
        }


class HandBlockEnv(ManipulateEnv):
    def __init__(self, target_position='random', target_rotation='xyz'):
        super(HandBlockEnv, self).__init__(
            model_path='hand/manipulate_block.xml', target_position=target_position,
            target_rotation=target_rotation, position_weight=25.,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]))


class HandEggEnv(ManipulateEnv):
    def __init__(self, target_position='random', target_rotation='xyz'):
        super(HandEggEnv, self).__init__(
            model_path='hand/manipulate_egg.xml', target_position=target_position,
            target_rotation=target_rotation, position_weight=25.,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]))


class HandPenEnv(ManipulateEnv):
    def __init__(self, target_position='random', target_rotation='xyz'):
        initial_qpos = {
            'object:rx': 1.9500000000000015,
            'object:ry': 1.9500000000000015,
            'object:rz': 0.7983724628009656,
        }
        super(HandPenEnv, self).__init__(
            model_path='hand/manipulate_pen.xml', target_position=target_position,
            target_rotation=target_rotation, position_weight=25.,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            initial_qpos=initial_qpos, randomize_initial_rotation=False)
