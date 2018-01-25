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


class ReachEnv(hand_env.HandEnv, utils.EzPickle):
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
        hand_env.HandEnv.__init__(
            self, 'reach.xml', n_substeps=20, initial_qpos=initial_qpos, relative_control=False,
            dist_threshold=0.02)
        utils.EzPickle.__init__(self)

    def _get_initial_state(self):
        super(ReachEnv, self)._get_initial_state()
        self.initial_goal = self._get_achieved_goal().copy()
        self.palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:palm')].copy()

    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for name in hand_env.FINGERTIP_SITE_NAMES]
        return np.array(goal)

    def _get_obs(self):
        robot_qpos, robot_qvel = hand_env.robot_get_obs(self.sim)
        achieved_goal = self._get_achieved_goal().flatten()
        observation = np.concatenate([robot_qpos, robot_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'goal': self.goal.flatten().copy(),
        }

    def _sample_goal(self):
        thumb_name = 'robot0:S_thtip'
        finger_names = [name for name in hand_env.FINGERTIP_SITE_NAMES if name != thumb_name]
        finger_name = np.random.choice(finger_names)

        thumb_idx = hand_env.FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx = hand_env.FINGERTIP_SITE_NAMES.index(finger_name)
        assert thumb_idx != finger_idx

        # Pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
        meeting_pos += np.random.normal(scale=0.005, size=meeting_pos.shape)

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
        return goal

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


