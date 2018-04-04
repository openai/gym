"""
A variation on the cart-pole model, where the base of the pole attached
to a fixed pivot point on the ground, and a weighted cart moves along
a table attached to the top of the pole.
"""

import math

import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

class CartTableEnv(gym.Env):
    """
    State:

        x = distance of cart from center of pole
        x_dot = velocity of cart moving away or towards the pole
        theta = angle of pole from the ground
        theta_dot = angular velocity of the pole
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length) # kg * meter
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2) # only two actions, move left or right
        self.observation_space = spaces.Box(-high, high) # pylint: disable=invalid-unary-operand-type

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state

        if action == 0:
            force = self.force_mag
        else:
            force = -self.force_mag

        # force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Calculate the initial 2D position (x, pole_length) of cart due to theta.
        xp = x*math.cos(theta) - self.length*math.sin(theta)
        yp = x*math.sin(theta) + self.length*math.cos(theta)

        # Calculate angle of cart offset from table center.
        theta_p = math.atan(x/float(self.length))

        # Calculate total angle of virtual arm from pivot point to the cart.
        arm_theta = theta + theta_p

        # Calculate length of virtual arm.
        arm_length = math.sqrt(xp**2 + yp**2)

        # Calculate torque of cart acting on virtual arm.
        # torque = radius * force * sin(theta)
        arm_torque = arm_length * self.masscart * math.sin(arm_theta)

        # Calculate table's angular acceleration due to torque.
        # torque = moment * angular acceleration => angular acceleration = torque / moment
        # If we treat the cart as a point mass rotating around the pivot => I = m*r^2
        thetaacc = arm_torque/(self.masscart * arm_length**2)

        # Calculate acceleration of the cart due to applied force.
        # a = (V1 - V0)/(t1 - t0) = F/m
        xacc = force * self.masscart #TODO:include gravity component as table tilts?

        # Calculate change in the cart's position due to velocity over time.
        # Df = Di + t * V
        x = x + self.tau * x_dot

        # Calculate change in cart's velocity due to acceleration over time.
        # Vf = Vi + t * a
        x_dot = x_dot + self.tau * xacc

        # Calculate change in pole's angle due to angular velocity over time.
        theta = theta + self.tau * theta_dot

        # Calculate change in pole's angular velocity due to angular acceleration over time.
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done = x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            self.trans = rendering.Transform()

            # Draw pole.
            axleoffset = cartheight/4.0
            l, r, t, b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.trans)
            self.viewer.add_geom(pole)

            # Draw pole axle.
            axle = rendering.make_circle(polewidth/2)
            axle.add_attr(self.poletrans)
            axle.add_attr(self.trans)
            axle.set_color(.5, .5, .8)
            self.viewer.add_geom(axle)

            # Draw cart.
            l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.add_attr(self.poletrans)
            cart.add_attr(self.trans)
            self.viewer.add_geom(cart)

            # Draw table.
            tl = cartwidth*2.5
            tt = 3
            table = rendering.FilledPolygon([(-tl, -tt), (-tl, tt), (tl, tt), (tl, -tt)])
            table.set_color(.8, .6, .4)
            carttrans = rendering.Transform(translation=(0, polelen-2))
            table.add_attr(carttrans)
            table.add_attr(self.poletrans)
            table.add_attr(self.trans)
            self.viewer.add_geom(table)

            # Draw track.
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        self.carttrans.set_translation(x[0]*scale, polelen+cartheight/2)
        cartx = screen_width/2.0 # MIDDLE OF THE SCREEN
        self.trans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
