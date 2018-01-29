import numpy as np

from gym import utils, error
from gym.envs.robotics import rotations, hand_env
from gym.envs.robotics.utils import robot_get_obs

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def goal_distance(goal_a, goal_b, pos_mul, compute_rotation):
    assert goal_a.shape == goal_b.shape
    assert goal_a.shape[-1] == 6
            
    diff = goal_a - goal_b
    diff[..., :3] *= pos_mul  # position
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
    def __init__(self, model_path, target_pos, target_rot, pos_mul, target_pos_range, initial_qpos={},
        randomize_initial_pos=True, randomize_initial_rot=True):
        self.target_pos = target_pos
        self.target_rot = target_rot
        self.pos_mul = pos_mul
        self.target_pos_range = target_pos_range
        self.parallel_rotations = rotations.get_parallel_rotations()
        self.randomize_initial_rot = randomize_initial_rot
        self.randomize_initial_pos = randomize_initial_pos
        self.dist_threshold = 0.4

        hand_env.HandEnv.__init__(
            self, model_path, n_substeps=20, initial_qpos=initial_qpos, relative_control=False)
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
            achieved_goal, goal, pos_mul=self.pos_mul,
            compute_rotation=self.target_rot != 'ignore')
        return -(d > self.dist_threshold).astype(np.float32)

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, goal):
        d = goal_distance(
            achieved_goal, goal, pos_mul=self.pos_mul,
            compute_rotation=self.target_rot != 'ignore')
        return (d < self.dist_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = get_block_qpos(self.sim, self.initial_state.qpos).copy()

        # Randomization initial rotation.
        if self.randomize_initial_rot:
            uniform_rot = np.random.uniform(0.0, 2 * np.pi, size=(3,))
            if self.target_rot == 'z':
                initial_qpos[3:] = rotations.subtract_euler(np.concatenate([np.zeros(2), [uniform_rot[2]]]), initial_qpos[3:])
            elif self.target_rot in ['xyz', 'parallel', 'ignore']:
                initial_qpos[3:] = uniform_rot
            elif self.target_rot == 'fixed':
                pass
            else:
                raise error.Error('Unknown target_rot option "{}".'.format(self.target_rot))

        # Randomize initial position.
        if self.randomize_initial_pos:
            if self.target_pos != 'fixed':
                initial_qpos[:3] += np.random.normal(size=3, scale=0.005)

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
        target_pos = np.zeros(3)
        if self.target_pos == 'random':
            assert self.target_pos_range.shape == (3, 2)
            offset = np.random.uniform(self.target_pos_range[:, 0], self.target_pos_range[:, 1])
            assert offset.shape == (3,)
            target_pos = get_block_qpos(self.sim, self.sim.data.qpos)[:3] + offset
        elif self.target_pos == 'ignore' or self.pos_mul == 0.:
            target_pos[:] = 0.
        elif self.target_pos == 'fixed':
            target_pos = get_block_qpos(self.sim, self.sim.data.qpos)[:3]
        else:
            raise error.Error('Unknown target_pos option "{}".'.format(self.target_pos))
        assert target_pos.shape == (3,)

        # Select a goal for the block rotation.
        target_rot = np.zeros(3)
        if self.target_rot == 'z':
            target_rot[-1] = np.random.uniform(-np.pi, np.pi)
        elif self.target_rot == 'parallel':
            target_rot[-1] = np.random.uniform(-np.pi, np.pi)
            target_rot[:2] = 0.
            parallel_rot = self.parallel_rotations[np.random.randint(len(self.parallel_rotations))]
            target_rot = rotations.mat2euler(np.matmul(rotations.euler2mat(target_rot), rotations.euler2mat(parallel_rot)))
        elif self.target_rot == 'xyz':
            target_rot = np.random.uniform(-np.pi, np.pi, size=3)
        elif self.target_rot == 'ignore':
            target_rot[:] = 0.
        elif self.target_rot == 'fixed':
            target_rot = get_block_qpos(self.sim, self.sim.data.qpos)[3:]
        else:
            raise error.Error('Unknown target_rot option "{}".'.format(self.target_rot))
        assert target_rot.shape == (3,)

        goal = np.concatenate([target_pos, rotations.normalize_angles(target_rot)])
        return goal

    def _render_callback(self):
        joint_names_pos = ['target:tx', 'target:ty', 'target:tz']
        joint_names_rot = ['target:rx', 'target:ry', 'target:rz']

        # Assign current state to target block but offset a bit so that the actual block
        # is not obscured.
        goal = self.goal.copy()
        if self.target_pos == 'ignore':
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
    def __init__(self, target_pos='random', target_rot='xyz'):
        super(HandBlockEnv, self).__init__(
            model_path='hand/manipulate_block.xml', target_pos=target_pos, target_rot=target_rot,
            pos_mul=25., target_pos_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]))


class HandEggEnv(ManipulateEnv):
    def __init__(self, target_pos='random', target_rot='xyz'):
        super(HandEggEnv, self).__init__(
            model_path='hand/manipulate_egg.xml', target_pos=target_pos, target_rot=target_rot,
            pos_mul=25., target_pos_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]))


class HandPenEnv(ManipulateEnv):
    def __init__(self, target_pos='random', target_rot='xyz'):
        initial_qpos = {
            'object:rx': 1.9500000000000015,
            'object:ry': 1.9500000000000015,
            'object:rz': 0.7983724628009656,
        }
        super(HandPenEnv, self).__init__(
            model_path='hand/manipulate_pen.xml', target_pos=target_pos, target_rot=target_rot,
            pos_mul=25., target_pos_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            initial_qpos=initial_qpos, randomize_initial_rot=False)
