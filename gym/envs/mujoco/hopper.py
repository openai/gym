import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment is based on the work done by Erez, Tassa, and Todorov in
    ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"]
    (http://www.roboticsproceedings.org/rss07/p10.pdf). The environment aims to
    increase the number of independent state and control variables as compared to
    the classic control environments. The hopper is a two-dimensional
    one-legged figure that consist of four main body parts - the torso at the
    top, the thigh in the middle, the leg in the bottom, and a single foot on
    which the entire body rests. The goal is to make hops that move in the
    forward (right) direction by applying torques on the three hinges
    connecting the four body parts.

    ### Action Space
    The agent take a 3-element vector for actions.
    The action space is a continuous `(action, action, action)` all in `[-1, 1]`
    , where `action` represents the numerical torques applied between *links*

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
    |-------|----------------------|---------------|----------------|---------------------------------------|-------|------|
    | 0   | Torque applied on the thigh rotor | -1 | 1 | thigh_joint  | hinge | torque (N m) |
    | 1   | Torque applied on the leg rotor    | -1 | 1 | leg_joint     | hinge | torque (N m) |
    | 3   | Torque applied on the foot rotor  | -1 | 1 | foot_joint    | hinge | torque (N m) |

    ### Observation Space

    The state space consists of positional values of different body parts of the
    hopper, followed by the velocities of those individual parts
    (their derivatives) with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(11,)` where the elements
    correspond to the following:

    | Num | Observation           | Min                  | Max                | Name (in corresponding XML file) | Joint| Unit |
    |-----|-----------------------|----------------------|--------------------|----------------------|--------------------|--------------------|
    | 0       | x-coordinate of the top                                    | -Inf                 | Inf                | rootx | slide | position (m) |
    | 1       | z-coordinate of the top (height of hopper)       | -Inf                 | Inf                | rootz | slide | position (m) |
    | 2       | angle of the top                                                | -Inf                 | Inf                | rooty | hinge | angle (rad) |
    | 3       | angle of the thigh joint                                      | -Inf                 | Inf                | thigh_joint | hinge | angle (rad) |
    | 4       | angle of the leg joint                                         | -Inf                 | Inf                | leg_joint | hinge | angle (rad) |
    | 5       | angle of the foot joint                                        | -Inf                 | Inf                | foot_joint | hinge | angle (rad) |
    | 6       | velocity of the x-coordinate of the top              | -Inf                 | Inf                | rootx | slide | velocity (m/s) |
    | 7       | velocity of the z-coordinate (height) of the top | -Inf                 | Inf                | rootz | slide | velocity (m/s)  |
    | 8       | angular velocity of the angle of the top            | -Inf                 | Inf                | rooty | hinge | angular velocity (rad/s) |
    | 9       | angular velocity of the thigh hinge                   | -Inf                 | Inf                | thigh_joint | hinge | angular velocity (rad/s) |
    | 10     | angular velocity of the leg hinge                       | -Inf                 | Inf                | leg_joint | hinge | angular velocity (rad/s) |
    | 11     | angular velocity of the foot hinge                     | -Inf                 | Inf                | foot_joint | hinge | angular velocity (rad/s) |



    **Note:**
    In practice (and Gym implementation), the first positional element is
    omitted from the state space since the reward function is calculated based
    on that value. This value is hidden from the algorithm, which in turn has
    to develop an abstract understanding of it from the observed rewards.
    Therefore, observation space has shape `(11,)` instead of `(12,)` and looks like:
    | Num | Observation           | Min                  | Max                | Name (in corresponding XML file) | Joint| Unit |
    |-----|-----------------------|----------------------|--------------------|----------------------|--------------------|--------------------|
    | 0       | z-coordinate of the top (height of hopper)        | -Inf                 | Inf                | rootz | slide | position (m) |
    | 1       | angle of the top                                                 | -Inf                 | Inf                | rooty | hinge | angle (rad) |
    | 2       | angle of the thigh joint                                       | -Inf                 | Inf                | thigh_joint | hinge | angle (rad) |
    | 3       | angle of the leg joint                                          | -Inf                 | Inf                | leg_joint | hinge | angle (rad) |
    | 4       | angle of the foot joint                                         | -Inf                 | Inf                | foot_joint | hinge | angle (rad) |
    | 5       | velocity of the x-coordinate of the top               | -Inf                 | Inf                | rootx | slide | velocity (m/s) |
    | 6       | velocity of the z-coordinate (height) of the top  | -Inf                 | Inf                | rootz | slide | velocity (m/s)  |
    | 7       | angular velocity of the angle of the top              | -Inf                 | Inf                | rooty | hinge | angular velocity (rad/s) |
    | 8       | angular velocity of the thigh hinge                      | -Inf                 | Inf                | thigh_joint | hinge | angular velocity (rad/s) |
    | 9       | angular velocity of the leg hinge                         | -Inf                 | Inf                | leg_joint | hinge | angular velocity (rad/s) |
    | 10     | angular velocity of the foot hinge                       | -Inf                 | Inf                | foot_joint | hinge | angular velocity (rad/s) |

    ### Rewards
    The reward consists of three parts:
    - *alive bonus*: Every timestep that the hopper is alive, it gets a reward of 1,
    - *reward_forward*: A reward of hopping forward which is measured
    as *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is
    the time between actions and is dependeent on the frame_skip parameter
    (default is 4), where the *dt* for one frame is 0.002 - making the
    default *dt = 4*0.002 = 0.008*. This reward would be positive if the hopper
    hops forward (right) desired.
    - *reward_control*: A negative reward for penalising the hopper if it takes
    actions that are too large. It is measured as *-coefficient **x**
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.001

    The total reward returned is ***reward*** *=* *alive bonus + reward_forward + reward_control*

    ### Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform nois
    e in the range of [-0.005, 0.005] added to the values for stochasticity.

    ### Episode Termination
    The episode terminates when any of the following happens:

    1. The episode duration reaches a 1000 timesteps
    2. Any of the state space values is no longer finite
    3. The absolute value of any of the state variable indexed (angle and beyond) is greater than 100
    4. The height of the hopper becomes greater than 0.7 metres (hopper has hopped too high).
    5. The absolute value of the angle (index 2) is less than 0.2 radians (hopper has fallen down).

    ### Arguments

    No additional arguments are currently supported (in v2 and lower), but
    modifications can be made to the XML file in the assets folder
    (or by changing the path to a modified XML file in another folder).

    ```
    env = gym.make('Hopper-v2')
    ```

    v3 and beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

    ```
    env = gym.make('Hopper-v3', ctrl_cost_weight=0.1, ....)
    ```

    ### Version History

    * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """
    
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "hopper.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
