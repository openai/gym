import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_box, target_in_the_air, target_x_shift, obj_range, target_range,
        dist_threshold, initial_qpos
    ):
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_box = has_box
        self.target_in_the_air = target_in_the_air
        self.target_x_shift = target_x_shift
        self.obj_range = obj_range
        self.target_range = target_range
        self.dist_threshold = dist_threshold

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        return -(d > self.dist_threshold).astype(np.float32)

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        gripper_ctrl = action[3]
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([action[:3], [1., 0., 1., 0.]])
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([action, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_box:
            box_pos = self.sim.data.get_site_xpos('geom0')
            # rotations
            box_rot = rotations.mat2euler(self.sim.data.get_site_xmat('geom0'))
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
        obs = np.concatenate([
            grip_pos, box_rel_pos.flatten(), gripper_state, box_rot.flatten(),
            box_velp.flatten(), box_velr.flatten(), grip_velp, gripper_vel,
        ])

        if not self.has_box:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(box_pos.copy())

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'goal': self.goal.copy(),
        }

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]

    def _sample_goal(self):
        if not self.has_box:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        else:
            box_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(box_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                box_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal[0] += self.target_x_shift
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
            qpos = self.initial_state.qpos
            qpos[-6:-4] = box_xpos
            qpos[-3:] = 0.  # no rotation
        return goal.copy()

    def _is_success(self, achieved_goal, goal):
        d = goal_distance(achieved_goal, goal)
        return (d < self.dist_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_box:
            self.height_offset = self.sim.data.get_site_xpos('geom0')[2]
