__credits__ = ["Rushiv Arora"]

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment corresponds to the Swimmer environment described in Rémi Coulom's PhD thesis ["Reinforcement Learning Using Neural Networks, with Applications to Motor Control"](https://tel.archives-ouvertes.fr/tel-00003985/document). The environment aims to increase the number of indepedent state and control variables as compared to the classic control environments. The swimmers consist of three or more segments ('***links***') and one less articulation joints ('***rotors***') - one rotor joint connecting exactly two links to form a linear chain. The swimmer is suspended in a two dimensional pool and always starts in the same position (subject to some deviation drawn from an uniform distribution), and the goal is to move as fast as possible towards the right by applying torque on the rotors and using the fluids friction.

    ### Notes

    The problem parameters are:
    Problem parameters:
    * *n*: number of body parts
    * *m<sub>i</sub>*: mass of part *i* (*i* ∈ {1...n}) 
    * *l<sub>i</sub>*: length of part *i* (*i* ∈ {1...n}) 
    * *k*: viscous-friction coefficient

    While the default environment has *n* = 3, *l<sub>i</sub>* = 0.1, and *k* = 0.1. It is possible to tweak the MuJoCo XML files to increase the number of links, or to tweak any of the parameters.

    ### Action Space
    The agent take a 2-element vector for actions.
    The action space is a continuous `(action, action)` in `[-1, 1]`, where `action` represents the numerical torques applied between *links*

    | Num | Action                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
    |-----|------------------------|-----|-----|------|-----|-----|
    | 0   | Torque applied on the first rotor  | -1 | 1 | rot2 | hinge | torque (N m) |
    | 1   | Torque applied on the second rotor  | -1 | 1 | rot3 | hinge | torque (N m) |

    ### Observation Space

    The state space consists of:
    * A<sub>0</sub>: position of first point
    * θ<sub>i</sub>: angle of part *i* with respect to the *x* axis
    * A<sub>0</sub>, θ<sub>i</sub>: their derivatives with respect to time (velocity and angular velocity)

    The observation is a `ndarray` with shape `(8,)` where the elements correspond to the following:

    | Num | Observation           | Min                  | Max                | Name (in corresponding XML file) | Joint| Unit |
    |-----|-----------------------|----------------------|--------------------|----------------------|--------------------|--------------------|
    | 0   | x-coordinate of the front tip              | -Inf                 | Inf                | slider1 | slide | position (m) |
    | 1   | y-coordinate of the front tip              | -Inf                 | Inf                | slider2 | slide | position (m) |
    | 2   | angle of the front tip                          | -Inf                 | Inf                | rot | hinge | angle (rad) |
    | 3   | angle of the second rotor                  | -Inf                 | Inf                | rot2 | hinge | angle (rad) |
    | 4   | angle of the second rotor                  | -Inf                 | Inf                | rot3 | hinge | angle (rad) |
    | 5   | velocity of the tip along the x-axis    | -Inf                 | Inf                | slider1 | slide | velocity (m/s) |
    | 6   | velocity of the tip along the y-axis    | -Inf                 | Inf                | slider2 | slide | velocity (m/s) |
    | 7   | angular velocity of front tip               | -Inf                 | Inf                | rot | hinge | angular velocity (rad/s) |
    | 8   | angular velocity of second rotor       | -Inf                 | Inf                | rot2 | hinge | angular velocity (rad/s) |
    | 9   | angular velocity of third rotor            | -Inf                 | Inf                | rot3 | hinge | angular velocity (rad/s) |

    **Note:**
    In practice (and Gym implementation), the first two positional elements are omitted from the state space since the reward function is calculated based on those values. Therefore, observation space has shape `(8,)` and looks like:
    | Num | Observation           | Min                  | Max                | Name (in corresponding XML file) | Joint| Unit |
    |-----|-----------------------|----------------------|--------------------|----------------------|--------------------|--------------------|
    | 0   | angle of the front tip                          | -Inf                 | Inf                | rot | hinge | angle (rad) |
    | 1   | angle of the second rotor                  | -Inf                 | Inf                | rot2 | hinge | angle (rad) |
    | 2   | angle of the second rotor                  | -Inf                 | Inf                | rot3 | hinge | angle (rad) |
    | 3   | velocity of the tip along the x-axis    | -Inf                 | Inf                | slider1 | slide | velocity (m/s) |
    | 4   | velocity of the tip along the y-axis    | -Inf                 | Inf                | slider2 | slide | velocity (m/s) |
    | 5   | angular velocity of front tip               | -Inf                 | Inf                | rot | hinge | angular velocity (rad/s) |
    | 6   | angular velocity of second rotor       | -Inf                 | Inf                | rot2 | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of third rotor            | -Inf                 | Inf                | rot3 | hinge | angular velocity (rad/s) |

    ### Rewards
    The reward consists of two parts:
    - *reward_front*: A reward of moving forward which is measured as *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the time between actions and is dependeent on tthe frame_skip parameter (default is 4), where the *dt* for one frame is 0.01 - making the default *dt = 4*0.01 = 0.04*. This reward would be positive if the swimmer swims right as desired.
    - *reward_control*: A negative reward for penalising the swimmer if it takes actions that are too large. It is measured as *-coefficient x sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the control and has a default value of 0.0001

    The total reward returned is ***reward*** *=* *reward_front + reward_control*

    ### Starting State
    All observations start in state (0,0,0,0,0,0,0,0) with a Uniform noise in the range of [-0.1, 0.1] is added to the initial state for stochasticity.

    ### Episode Termination
    The episode terminates when the episode length is greater than 1000.

    ### Arguments

    No additional arguments are currently supported (in v2 and lower), but modifications can be made to the XML file in the assets folder (or by changing the path to a modified XML file in another folder)..

    ```
    gym.make('Swimmer-v2')
    ```

    v3 and beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

    ```
    env = gym.make('Swimmer-v3', ctrl_cost_weight=0.1, ....)
    ```

    ### Version History

    * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "swimmer.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()
