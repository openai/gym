import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment builds on the hopper environment based on the work done by Erez, Tassa, and Todorov
    in ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf)
    by adding another set of legs making it possible for the robot to walker forward instead of
    hop. Like other Mujoco environments, this environment aims to increase the number of independent state
    and control variables as compared to the classic control environments. The walker is a
    two-dimensional two-legged figure that consist of four main body parts - a single torso at the top
    (with the two legs splitting after the torso), two thighs in the middle below the torso, two legs
    in the bottom below the thighs, and two feet attached to the legs on which the entire body rests.
    The goal is to make coordinate both sets of feet, legs, and thighs to move in the forward (right)
    direction by applying torques on the six hinges connecting the six body parts.

    ### Action Space
    The agent take a 6-element vector for actions.
    The action space is a continuous `(action, action, action, action, action, action)`
    all in `[-1, 1]`, where `action` represents the numerical torques applied at the hinge joints.

    | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
    | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
    | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
    | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
    | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
    | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |

    ### Observation Space

    The state space consists of positional values of different body parts of the walker,
    followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(17,)` where the elements correspond to the following:

    | Num | Observation                                      | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    |-----|--------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
    | 0   | x-coordinate of the top                          | -Inf | Inf | rootx (torso)                    | slide | position (m)             |
    | 1   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz (torso)                    | slide | position (m)             |
    | 2   | angle of the top                                 | -Inf | Inf | rooty (torso)                    | hinge | angle (rad)              |
    | 3   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
    | 4   | angle of the leg joint                           | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
    | 5   | angle of the foot joint                          | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
    | 6   | angle of the left thigh joint                    | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
    | 7   | angle of the left leg joint                      | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
    | 8   | angle of the left foot joint                     | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
    | 9   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 10  | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 11  | angular velocity of the angle of the top         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 12  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of the leg hinge                | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
    | 17  | angular velocity of the foot hinge               | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |

    **Note:**
    In practice (and Gym implementation), the first positional element is omitted from the
    state space since the reward function is calculated based on that value. This value is
    hidden from the algorithm, which in turn has to develop an abstract understanding of it
    from the observed rewards. Therefore, observation space has shape `(17,)`
    instead of `(18,)` and looks like:

    | Num | Observation                                      | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    |-----|--------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
    | 0   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz (torso)                    | slide | position (m)             |
    | 1   | angle of the top                                 | -Inf | Inf | rooty (torso)                    | hinge | angle (rad)              |
    | 2   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
    | 3   | angle of the leg joint                           | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
    | 4   | angle of the foot joint                          | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
    | 5   | angle of the left thigh joint                    | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
    | 6   | angle of the left leg joint                      | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
    | 7   | angle of the left foot joint                     | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
    | 8   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 9   | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 10  | angular velocity of the angle of the top         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 11  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
    | 12  | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of the leg hinge                | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of the foot hinge               | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |

    ### Rewards
    The reward consists of three parts:
    - *alive bonus*: Every timestep that the walker is alive, it gets a reward of 1,
    - *reward_forward*: A reward of walking forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*.
    *dt* is the time between actions and is dependent on the frame_skip parameter
    (default is 4), where the *dt* for one frame is 0.002 - making the default
    *dt = 4 * 0.002 = 0.008*. This reward would be positive if the walker walks forward (right) desired.
    - *reward_control*: A negative reward for penalising the walker if it
    takes actions that are too large. It is measured as
    *-coefficient **x** sum(action<sup>2</sup>)* where *coefficient* is
    a parameter set for the control and has a default value of 0.001

    The total reward returned is ***reward*** *=* *alive bonus + reward_forward + reward_control*

    ### Starting State
    All observations start in state
    (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with a uniform noise in the range of [-0.005, 0.005] added to the values for stochasticity.

    ### Episode Termination
    The episode terminates when any of the following happens:

    1. The episode duration reaches a 1000 timesteps
    2. Any of the state space values is no longer finite
    3. The height of the walker (index 1) is ***not*** in the range `[0.8, 2]`
    4. The absolute value of the angle (index 2) is ***not*** in the range `[-1, 1]`

    ### Arguments

    No additional arguments are currently supported (in v2 and lower),
    but modifications can be made to the XML file in the assets folder
    (or by changing the path to a modified XML file in another folder)..

    ```
    env = gym.make('Walker2d-v2')
    ```

    v3 and v4 take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

    ```
    env = gym.make('Walker2d-v4', ctrl_cost_weight=0.1, ....)
    ```

    ### Version History

    * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
    * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)

    """

    def __init__(
        self,
        xml_file="walker2d.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        mujoco_env.MujocoEnv.__init__(self, xml_file, 4)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
