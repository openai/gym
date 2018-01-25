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



class BlockEnv(hand_env.HandEnv, utils.EzPickle):
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
        hand_env.HandEnv.__init__(
            self, 'block.xml', n_substeps=20, initial_qpos=initial_qpos, relative_control=False,
            dist_threshold=0.4)
        utils.EzPickle.__init__(self)

    def _reset_simulation(self):
        super(BlockEnv, self)._reset_simulation()

        # randomize rot
        initial_qpos = self._get_block_qpos(self.initial_state.qpos).copy()
        uniform_rot = np.random.uniform(0.0, 2 * np.pi, size=(3,))
        if self.target_rot == 'z':
            initial_qpos[3:] = rotations.subtract_euler(np.concatenate([np.zeros(2), [uniform_rot[2]]]), initial_qpos[3:])
        elif self.target_rot in ['xyz', 'parallel', 'ignore']:
            initial_qpos[3:] = uniform_rot
        elif self.target_rot == 'fixed':
            pass
        else:
            raise error.Error('Unknown target_rot option "{}".'.format(self.target_rot))

        # randomize pos
        if self.target_pos != 'fixed':
            initial_qpos[:3] += np.random.normal(size=3, scale=0.005)

        self._set_block_qpos(initial_qpos)

        def is_on_palm():
            self.sim.forward()
            cube_middle_idx = self.sim.model.site_name2id('block:center')
            cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(100):
            hand_env.set_action(self.sim, np.zeros(20))
            self.sim.step()
        assert is_on_palm()

    def _get_block_qpos(self, qpos):
        joint_names = ['block_tx', 'block_ty', 'block_tz', 'block_rx', 'block_ry', 'block_rz']
        block_qpos = []
        for name in joint_names:
            addr = self.sim.model.get_joint_qpos_addr('block:{}'.format(name))
            block_qpos.append(qpos[addr])
        block_qpos = np.array(block_qpos)
        assert block_qpos.shape == (6,)
        return block_qpos

    def _set_block_qpos(self, qpos):
        assert qpos.shape == (6,)
        joint_names = ['block_tx', 'block_ty', 'block_tz', 'block_rx', 'block_ry', 'block_rz']
        for name, value in zip(joint_names, qpos):
            addr = self.sim.model.get_joint_qpos_addr('block:{}'.format(name))
            self.sim.data.qpos[addr] = value

    def _get_block_qvel(self, qvel):
        joint_names = ['block_tx', 'block_ty', 'block_tz', 'block_rx', 'block_ry', 'block_rz']
        block_qvel = []
        for name in joint_names:
            addr = self.sim.model.get_joint_qvel_addr('block:{}'.format(name))
            block_qvel.append(qvel[addr])
        block_qvel = np.array(block_qvel)
        assert block_qvel.shape == (6,)
        return block_qvel

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


class BlockRotateXYZEnv(BlockEnv):
    def __init__(self):
        super(BlockRotateXYZEnv, self).__init__(
            target_pos='ignore', target_rot='xyz', pos_mul=0.)


class BlockRotateZEnv(BlockEnv):
    def __init__(self):
        super(BlockRotateZEnv, self).__init__(
            target_pos='ignore', target_rot='z', pos_mul=0.)
