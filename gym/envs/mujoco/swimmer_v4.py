__credits__ = ["Rushiv Arora"]

import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

DEFAULT_CAMERA_CONFIG = {}


class SwimmerEnv(MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment corresponds to the Swimmer environment described in Rémi Coulom's PhD thesis
    ["Reinforcement Learning Using Neural Networks, with Applications to Motor Control"](https://tel.archives-ouvertes.fr/tel-00003985/document).
    The environment aims to increase the number of independent state and control
    variables as compared to the classic control environments. The swimmers
    consist of three or more segments ('***links***') and one less articulation
    joints ('***rotors***') - one rotor joint connecting exactly two links to
    form a linear chain. The swimmer is suspended in a two dimensional pool and
    always starts in the same position (subject to some deviation drawn from an
    uniform distribution), and the goal is to move as fast as possible towards
    the right by applying torque on the rotors and using the fluids friction.

    ### Notes

    The problem parameters are:
    Problem parameters:
    * *n*: number of body parts
    * *m<sub>i</sub>*: mass of part *i* (*i* ∈ {1...n})
    * *l<sub>i</sub>*: length of part *i* (*i* ∈ {1...n})
    * *k*: viscous-friction coefficient

    While the default environment has *n* = 3, *l<sub>i</sub>* = 0.1,
    and *k* = 0.1. It is possible to pass a custom MuJoCo XML file during construction to increase the
    number of links, or to tweak any of the parameters.

    ### Action Space
    The action space is a `Box(-1, 1, (2,), float32)`. An action represents the torques applied between *links*

    | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the first rotor  | -1          | 1           | rot2                             | hinge | torque (N m) |
    | 1   | Torque applied on the second rotor | -1          | 1           | rot3                             | hinge | torque (N m) |

    ### Observation Space

    By default, observations consists of:
    * θ<sub>i</sub>: angle of part *i* with respect to the *x* axis
    * θ<sub>i</sub>': its derivative with respect to time (angular velocity)

    In the default case, observations do not include the x- and y-coordinates of the front tip. These may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    Then, the observation space will have 10 dimensions where the first two dimensions
    represent the x- and y-coordinates of the front tip.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x- and y-coordinates
    will be returned in `info` with keys `"x_position"` and `"y_position"`, respectively.

    By default, the observation is a `ndarray` with shape `(8,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | angle of the front tip               | -Inf | Inf | rot                              | hinge | angle (rad)              |
    | 1   | angle of the first rotor             | -Inf | Inf | rot2                             | hinge | angle (rad)              |
    | 2   | angle of the second rotor            | -Inf | Inf | rot3                             | hinge | angle (rad)              |
    | 3   | velocity of the tip along the x-axis | -Inf | Inf | slider1                          | slide | velocity (m/s)           |
    | 4   | velocity of the tip along the y-axis | -Inf | Inf | slider2                          | slide | velocity (m/s)           |
    | 5   | angular velocity of front tip        | -Inf | Inf | rot                              | hinge | angular velocity (rad/s) |
    | 6   | angular velocity of first rotor      | -Inf | Inf | rot2                             | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of second rotor     | -Inf | Inf | rot3                             | hinge | angular velocity (rad/s) |

    ### Rewards
    The reward consists of two parts:
    - *forward_reward*: A reward of moving forward which is measured
    as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*. *dt* is
    the time between actions and is dependent on the frame_skip parameter
    (default is 4), where the frametime is 0.01 - making the
    default *dt = 4 * 0.01 = 0.04*. This reward would be positive if the swimmer
    swims right as desired.
    - *ctrl_cost*: A cost for penalising the swimmer if it takes
    actions that are too large. It is measured as *`ctrl_cost_weight` *
    sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is a parameter set for the
    control and has a default value of 1e-4

    The total reward returned is ***reward*** *=* *forward_reward - ctrl_cost* and `info` will also contain the individual reward terms

    ### Starting State
    All observations start in state (0,0,0,0,0,0,0,0) with a Uniform noise in the range of [-`reset_noise_scale`, `reset_noise_scale`] is added to the initial state for stochasticity.

    ### Episode End
    The episode truncates when the episode length is greater than 1000.

    ### Arguments

    No additional arguments are currently supported in v2 and lower.

    ```
    gym.make('Swimmer-v4')
    ```

    v3 and v4 take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

    ```
    env = gym.make('Swimmer-v4', ctrl_cost_weight=0.1, ....)
    ```

    | Parameter                                    | Type      | Default         | Description                                                                                                                                                               |
    | -------------------------------------------- | --------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"swimmer.xml"` | Path to a MuJoCo model                                                                                                                                                    |
    | `forward_reward_weight`                      | **float** | `1.0`           | Weight for _forward_reward_ term (see section on reward)                                                                                                                  |
    | `ctrl_cost_weight`                           | **float** | `1e-4`          | Weight for _ctrl_cost_ term (see section on reward)                                                                                                                       |
    | `reset_noise_scale`                          | **float** | `0.1`           | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                            |
    | `exclude_current_positions_from_observation` | **bool**  | `True`          | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |


    ### Version History

    * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
    * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
            )
        MujocoEnv.__init__(
            self, "swimmer.xml", 4, observation_space=observation_space, **kwargs
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self._forward_reward_weight * x_velocity

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        self.renderer.render_step()
        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation

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
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
