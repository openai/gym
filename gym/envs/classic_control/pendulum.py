__credits__ = ["Carlos Luis"]

from typing import Optional

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    """
    ### Description

     The inverted pendulum swingup problem is based on the classic problem in control theory. The system consists of a pendulum attached at one end to a fixed point, and the other end being free. The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an upright position, with its center of gravity right above the fixed point.

     The diagram below specifies the coordinate system used for the implementation of the pendulum's
     dynamic equations.

     ![Pendulum Coordinate System](./diagrams/pendulum.png)

     -  `x-y`: cartesian coordinates of the pendulum's end in meters.
     - `theta` : angle in radians.
     - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

     ### Action Space

     The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

     | Num | Action | Min  | Max |
     |-----|--------|------|-----|
     | 0   | Torque | -2.0 | 2.0 |


     ### Observation Space

     The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free end and its angular velocity.

     | Num | Observation      | Min  | Max |
     |-----|------------------|------|-----|
     | 0   | x = cos(theta)   | -1.0 | 1.0 |
     | 1   | y = sin(angle)   | -1.0 | 1.0 |
     | 2   | Angular Velocity | -8.0 | 8.0 |

     ### Rewards

     The reward function is defined as:

     *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

     where `$\theta$` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
     Based on the above equation, the minimum reward that can be obtained is *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*, while the maximum reward is zero (pendulum is
     upright with zero velocity and no torque applied).

     ### Starting State

     The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

     ### Episode Termination

     The episode terminates at 200 time steps.

     ### Arguments

     - `g`: acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics. The default value is g = 10.0 .

     ```
     gym.make('Pendulum-v1', g=9.81)
     ```

     ### Version History

     * v1: Simplify the math equations, no difference in behavior.
     * v0: Initial versions release (1.0.0)


    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.viewer = None

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.utils import pyglet_rendering

            self.viewer = pyglet_rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = pyglet_rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = pyglet_rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = pyglet_rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = pyglet_rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = pyglet_rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
