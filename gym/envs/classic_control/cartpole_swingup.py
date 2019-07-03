"""
Cart pole swing-up: modified version of:
https://github.com/hardmaru/estool/blob/master/custom_envs/cartpole_swingup.py
"""
import math
from collections import namedtuple

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


Physics = namedtuple("Physics", "gravity forcemag deltat friction")

State = namedtuple("State", "x_pos x_dot theta theta_dot")

Screen = namedtuple("Screen", "width height")

Cart = namedtuple("Cart", "width height mass")

Pole = namedtuple("Pole", "width length mass")


class CartPoleSwingUpEnv(gym.Env):
    """
    Description:
       A pole is attached by an un-actuated joint to a cart, which moves along a track.
       Unlike CartPoleEnv, friction is taken into account in the physics calculations.
       The pendulum starts (pointing down) upside down, and the goal is to swing it up
       and keep it upright by increasing and reducing the cart's velocity.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
    physics = Physics(gravity=9.82, forcemag=10.0, deltat=0.01, friction=0.1)
    cart = Cart(width=1 / 3, height=1 / 6, mass=0.5)
    pole = Pole(width=0.05, length=0.6, mass=0.5)
    x_threshold = 2.4  # Distance limit from the center

    def __init__(self):
        high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ]
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reward(self):
        reward_theta = (np.cos(self.state.theta) + 1.0) / 2.0
        reward_x = np.cos((self.state.x_pos / self.x_threshold) * (np.pi / 2.0))
        return reward_theta * reward_x

    def _terminal(self):
        x_pos = self.state.x_pos
        if x_pos < -self.x_threshold or x_pos > self.x_threshold:
            return True
        return False

    def _get_obs(self):
        x_pos, x_dot, theta, theta_dot = self.state
        return np.array([x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot])

    def step(self, action):
        # Valid action
        action = np.clip(action, -1.0, 1.0)[0]
        action *= self.physics.forcemag

        state = self.state
        physics = self.physics

        sin_theta = math.sin(state.theta)
        cos_theta = math.cos(state.theta)

        m_p_l = self.pole.mass * self.pole.length
        masstotal = self.cart.mass + self.pole.mass
        xdot_update = (
            -2 * m_p_l * (state.theta_dot ** 2) * sin_theta
            + 3 * self.pole.mass * physics.gravity * sin_theta * cos_theta
            + 4 * action
            - 4 * physics.friction * state.x_dot
        ) / (4 * masstotal - 3 * self.pole.mass * cos_theta ** 2)
        thetadot_update = (
            -3 * m_p_l * (state.theta_dot ** 2) * sin_theta * cos_theta
            + 6 * masstotal * physics.gravity * sin_theta
            + 6 * (action - physics.friction * state.x_dot) * cos_theta
        ) / (4 * self.pole.length * masstotal - 3 * m_p_l * cos_theta ** 2)

        self.state = State(
            x_pos=state.x_pos + state.x_dot * physics.deltat,
            theta=state.theta + state.theta_dot * physics.deltat,
            x_dot=state.x_dot + xdot_update * physics.deltat,
            theta_dot=state.theta_dot + thetadot_update * physics.deltat,
        )

        return self._get_obs(), self._reward(), self._terminal(), {}

    def reset(self):
        self.state = State(
            *self.np_random.normal(
                loc=np.array([0.0, 0.0, np.pi, 0.0]),
                scale=np.array([0.2, 0.2, 0.2, 0.2]),
            )
        )
        return self._get_obs()

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = CartPoleSwingUpViewer(self.cart, self.pole, world_width=5)

        if self.state is None:
            return None

        self.viewer.update(self.state, self.pole)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleSwingUpViewer:
    screen = Screen(width=600, height=400)

    def __init__(self, cart, pole, world_width):
        from gym.envs.classic_control import rendering

        self.world_width = world_width
        screen = self.screen
        scale = screen.width / self.world_width
        cartwidth, cartheight = scale * cart.width, scale * cart.height
        polewidth, polelength = scale * pole.width, scale * pole.length
        self.viewer = rendering.Viewer(screen.width, screen.height)
        self.transforms = {
            "cart": rendering.Transform(),
            "pole": rendering.Transform(translation=(0, 0)),
            "pole_bob": rendering.Transform(),
            "wheel_l": rendering.Transform(
                translation=(-cartwidth / 2, -cartheight / 2)
            ),
            "wheel_r": rendering.Transform(
                translation=(cartwidth / 2, -cartheight / 2)
            ),
        }

        self._init_track(rendering, cartheight)
        self._init_cart(rendering, cartwidth, cartheight)
        self._init_wheels(rendering, cartheight)
        self._init_pole(rendering, polewidth, polelength)
        self._init_axle(rendering, polewidth)
        # Make another circle on the top of the pole
        self._init_pole_bob(rendering, polewidth)

    def _init_track(self, rendering, cartheight):
        screen = self.screen
        carty = screen.height / 2
        track_height = carty - cartheight / 2 - cartheight / 4
        track = rendering.Line((0, track_height), (screen.width, track_height))
        track.set_color(0, 0, 0)
        self.viewer.add_geom(track)

    def _init_cart(self, rendering, cartwidth, cartheight):
        lef, rig, top, bot = (
            -cartwidth / 2,
            cartwidth / 2,
            cartheight / 2,
            -cartheight / 2,
        )
        cart = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
        cart.add_attr(self.transforms["cart"])
        cart.set_color(1, 0, 0)
        self.viewer.add_geom(cart)

    def _init_pole(self, rendering, polewidth, polelength):
        lef, rig, top, bot = (
            -polewidth / 2,
            polewidth / 2,
            polelength - polewidth / 2,
            -polewidth / 2,
        )
        pole = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
        pole.set_color(0, 0, 1)
        pole.add_attr(self.transforms["pole"])
        pole.add_attr(self.transforms["cart"])
        self.viewer.add_geom(pole)

    def _init_axle(self, rendering, polewidth):
        axle = rendering.make_circle(polewidth / 2)
        axle.add_attr(self.transforms["pole"])
        axle.add_attr(self.transforms["cart"])
        axle.set_color(0.1, 1, 1)
        self.viewer.add_geom(axle)

    def _init_pole_bob(self, rendering, polewidth):
        pole_bob = rendering.make_circle(polewidth / 2)
        pole_bob.add_attr(self.transforms["pole_bob"])
        pole_bob.add_attr(self.transforms["pole"])
        pole_bob.add_attr(self.transforms["cart"])
        pole_bob.set_color(0, 0, 0)
        self.viewer.add_geom(pole_bob)

    def _init_wheels(self, rendering, cartheight):
        wheel_l = rendering.make_circle(cartheight / 4)
        wheel_r = rendering.make_circle(cartheight / 4)
        wheel_l.add_attr(self.transforms["wheel_l"])
        wheel_l.add_attr(self.transforms["cart"])
        wheel_r.add_attr(self.transforms["wheel_r"])
        wheel_r.add_attr(self.transforms["cart"])
        wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
        wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
        self.viewer.add_geom(wheel_l)
        self.viewer.add_geom(wheel_r)

    def update(self, state, pole):
        screen = self.screen
        scale = screen.width / self.world_width

        cartx = state.x_pos * scale + screen.width / 2.0  # MIDDLE OF CART
        carty = screen.height / 2
        self.transforms["cart"].set_translation(cartx, carty)
        self.transforms["pole"].set_rotation(state.theta)
        self.transforms["pole_bob"].set_translation(
            -pole.length * np.sin(state.theta), pole.length * np.cos(state.theta)
        )

    def render(self, *args, **kwargs):
        return self.viewer.render(*args, **kwargs)

    def close(self):
        self.viewer.close()
