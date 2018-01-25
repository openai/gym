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

from gym.envs.hand import rotations, hand_env

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))



class ManipulationEnv(hand_env.HandEnv, utils.EzPickle):
    def __init__(self, model_path, target_pos, target_rot, pos_mul, pos_range, initial_qpos={},
        randomize_initial_pos=True, randomize_initial_rot=True):
        self.target_pos = target_pos
        self.target_rot = target_rot
        self.pos_mul = pos_mul
        self.pos_range = pos_range
        self.parallel_rotations = rotations.get_parallel_rotations()
        self.randomize_initial_rot = randomize_initial_rot
        self.randomize_initial_pos = randomize_initial_pos

        hand_env.HandEnv.__init__(
            self, model_path, n_substeps=20, initial_qpos=initial_qpos, relative_control=False,
            dist_threshold=0.4)
        utils.EzPickle.__init__(self)

    def _reset_simulation(self):
        super(ManipulationEnv, self)._reset_simulation()

        initial_qpos = self._get_block_qpos(self.initial_state.qpos).copy()
        
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

        self._set_block_qpos(initial_qpos)

        def is_on_palm():
            self.sim.forward()
            cube_middle_idx = self.sim.model.site_name2id('object:center')
            cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            hand_env.set_action(self.sim, np.zeros(20))
            self.sim.step()
        assert is_on_palm()

    def _get_block_qpos(self, qpos):
        joint_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        block_qpos = []
        for name in joint_names:
            addr = self.sim.model.get_joint_qpos_addr('object:{}'.format(name))
            block_qpos.append(qpos[addr])
        block_qpos = np.array(block_qpos)
        assert block_qpos.shape == (6,)
        return block_qpos

    def _set_block_qpos(self, qpos):
        assert qpos.shape == (6,)
        joint_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for name, value in zip(joint_names, qpos):
            addr = self.sim.model.get_joint_qpos_addr('object:{}'.format(name))
            self.sim.data.qpos[addr] = value

    def _get_block_qvel(self, qvel):
        joint_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        block_qvel = []
        for name in joint_names:
            addr = self.sim.model.get_joint_qvel_addr('object:{}'.format(name))
            block_qvel.append(qvel[addr])
        block_qvel = np.array(block_qvel)
        assert block_qvel.shape == (6,)
        return block_qvel

    def _reset_goal(self):
        # Select a goal for the block position.
        target_pos = np.zeros(3)
        if self.target_pos == 'random':
            assert self.pos_range.shape == (3, 2)
            target_pos = self._get_block_qpos(self.sim.data.qpos)[:3] + np.random.uniform(self.pos_range[:, 0], self.pos_range[:, 1])
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
            target_rot[-1] = np.random.uniform(-np.pi, np.pi)
            target_rot[:2] = 0.
            parallel_rot = self.parallel_rotations[np.random.randint(len(self.parallel_rotations))]
            target_rot = rotations.mat2euler(np.matmul(rotations.euler2mat(target_rot), rotations.euler2mat(parallel_rot)))
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

    def _get_obs(self):
        robot_qpos, robot_qvel = hand_env.robot_get_obs(self.sim)
        block_qvel = self._get_block_qvel(self.sim.data.qvel)
        achieved_goal = self._get_achieved_goal().flatten()  # this contains the block position + rotation
        observation = np.concatenate([robot_qpos, robot_qvel, block_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'goal': self.goal.flatten().copy(),
        }


class BlockRotateXYZEnv(ManipulationEnv):
    def __init__(self):
        super(BlockRotateXYZEnv, self).__init__(
            model_path='manipulation_block.xml', target_pos='ignore', target_rot='xyz',
            pos_mul=0., pos_range=None)


class BlockRotateZEnv(ManipulationEnv):
    def __init__(self):
        super(BlockRotateZEnv, self).__init__(
            model_path='manipulation_block.xml', target_pos='ignore', target_rot='z',
            pos_mul=0., pos_range=None)


class BlockRotateParallelEnv(ManipulationEnv):
    def __init__(self):
        super(BlockRotateParallelEnv, self).__init__(
            model_path='manipulation_block.xml', target_pos='ignore', target_rot='parallel',
            pos_mul=0., pos_range=None)


class BlockPositionEnv(ManipulationEnv):
    def __init__(self):
        super(BlockPositionEnv, self).__init__(
            model_path='manipulation_block.xml', target_pos='random', target_rot='fixed',
            pos_mul=25., pos_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]))


class BlockPositionAndRotateZEnv(ManipulationEnv):
    def __init__(self):
        super(BlockPositionAndRotateZEnv, self).__init__(
            model_path='manipulation_block.xml', target_pos='random', target_rot='z',
            pos_mul=25., pos_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]))


class BlockPositionAndRotateXYZEnv(ManipulationEnv):
    def __init__(self):
        super(BlockPositionAndRotateXYZEnv, self).__init__(
            model_path='manipulation_block.xml', target_pos='random', target_rot='xyz',
            pos_mul=25., pos_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]))


class PenRotationEnv(ManipulationEnv):
    def __init__(self):
        initial_qpos = {
            'object:rx': 1.9500000000000015,
            'object:ry': 1.9500000000000015,
            'object:rz': 0.7983724628009656,
        }
        super(PenRotationEnv, self).__init__(
            model_path='manipulation_pen.xml', target_pos='ignore', target_rot='xyz',
            pos_mul=0., pos_range=None, initial_qpos=initial_qpos, randomize_initial_rot=False)
