import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, distanceJointDef,
                      contactListener)
import gym
from gym import spaces
from gym.utils import seeding

"""

The objective of this environment is to land a rocket on a ship.

STATE VARIABLES
The state consists of the following variables:
    - x position
    - y position
    - angle
    - first leg ground contact indicator
    - second leg ground contact indicator
    - throttle
    - engine gimbal
If VEL_STATE is set to true, the velocities are included:
    - x velocity
    - y velocity
    - angular velocity
all state variables are roughly in the range [-1, 1]
    
CONTROL INPUTS
Discrete control inputs are:
    - gimbal left
    - gimbal right
    - throttle up
    - throttle down
    - use first control thruster
    - use second control thruster
    - no action
    
Continuous control inputs are:
    - gimbal (left/right)
    - throttle (up/down)
    - control thruster (left/right)

"""

CONTINUOUS = True
VEL_STATE = True  # Add velocity info to state
FPS = 60
SCALE_S = 0.35  # Temporal Scaling, lower is faster - adjust forces appropriately
INITIAL_RANDOM = 0.4  # Random scaling of initial velocity, higher is more difficult

START_HEIGHT = 1000.0
START_SPEED = 80.0

# ROCKET
MIN_THROTTLE = 0.4
GIMBAL_THRESHOLD = 0.4
MAIN_ENGINE_POWER = 1600 * SCALE_S
SIDE_ENGINE_POWER = 100 / FPS * SCALE_S

ROCKET_WIDTH = 3.66 * SCALE_S
ROCKET_HEIGHT = ROCKET_WIDTH / 3.7 * 47.9
ENGINE_HEIGHT = ROCKET_WIDTH * 0.5
ENGINE_WIDTH = ENGINE_HEIGHT * 0.7
THRUSTER_HEIGHT = ROCKET_HEIGHT * 0.86

# LEGS
LEG_LENGTH = ROCKET_WIDTH * 2.2
BASE_ANGLE = -0.27
SPRING_ANGLE = 0.27
LEG_AWAY = ROCKET_WIDTH / 2

# SHIP
SHIP_HEIGHT = ROCKET_WIDTH
SHIP_WIDTH = SHIP_HEIGHT * 40

# VIEWPORT
VIEWPORT_H = 1440
VIEWPORT_W = 1008
H = 1.1 * START_HEIGHT * SCALE_S
W = float(VIEWPORT_W) / VIEWPORT_H * H

# SMOKE FOR VISUALS
MAX_SMOKE_LIFETIME = 2 * FPS

MEAN = np.array([-0.034, -0.15, -0.016, 0.0024, 0.0024, 0.137,
                 - 0.02, -0.01, -0.8, 0.002])
VAR = np.sqrt(np.array([0.08, 0.33, 0.0073, 0.0023, 0.0023, 0.8,
                        0.085, 0.0088, 0.063, 0.076]))


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.water in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.lander in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.containers[0] in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.containers[1] in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True
        else:
            for i in range(2):
                if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class RocketLander(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None
        self.episode_number = 0

        self.world = Box2D.b2World()
        self.water = None
        self.lander = None
        self.engine = None
        self.ship = None
        self.legs = []

        high = np.array([1, 1, 1, 1, 1, 1, 1, np.inf, np.inf, np.inf], dtype=np.float32)
        low = -high
        if not VEL_STATE:
            high = high[0:7]
            low = low[0:7]

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if CONTINUOUS:
            self.action_space = spaces.Box(-1.0, +1.0, (3,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(7)

        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.water:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.water)
        self.water = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.ship)
        self.ship = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self.legs = []
        self.world.DestroyBody(self.containers[0])
        self.world.DestroyBody(self.containers[1])
        self.containers = []

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.throttle = 0
        self.gimbal = 0.0
        self.landed_ticks = 0
        self.stepnumber = 0
        self.smoke = []

        # self.terrainheigth = self.np_random.uniform(H / 20, H / 10)
        self.terrainheigth = H / 20
        self.shipheight = self.terrainheigth + SHIP_HEIGHT
        # ship_pos = self.np_random.uniform(0, SHIP_WIDTH / SCALE) + SHIP_WIDTH / SCALE
        ship_pos = W / 2
        self.helipad_x1 = ship_pos - SHIP_WIDTH / 2
        self.helipad_x2 = self.helipad_x1 + SHIP_WIDTH
        self.helipad_y = self.terrainheigth + SHIP_HEIGHT

        self.water = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((0, 0), (W, 0), (W, self.terrainheigth), (0, self.terrainheigth))),
                friction=0.1,
                restitution=0.0)
        )
        self.water.color1 = rgb(70, 96, 176)

        self.ship = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=((self.helipad_x1, self.terrainheigth),
                              (self.helipad_x2, self.terrainheigth),
                              (self.helipad_x2, self.terrainheigth + SHIP_HEIGHT),
                              (self.helipad_x1, self.terrainheigth + SHIP_HEIGHT))),
                friction=0.5,
                restitution=0.0)
        )

        self.containers = []
        for side in [-1, 1]:
            self.containers.append(self.world.CreateStaticBody(
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=((ship_pos + side * 0.95 * SHIP_WIDTH / 2, self.helipad_y),
                                  (ship_pos + side * 0.95 * SHIP_WIDTH / 2, self.helipad_y + SHIP_HEIGHT),
                                  (ship_pos + side * 0.95 * SHIP_WIDTH / 2 - side * SHIP_HEIGHT,
                                   self.helipad_y + SHIP_HEIGHT),
                                  (ship_pos + side * 0.95 * SHIP_WIDTH / 2 - side * SHIP_HEIGHT, self.helipad_y)
                                  )),
                    friction=0.2,
                    restitution=0.0)
            ))
            self.containers[-1].color1 = rgb(206, 206, 2)

        self.ship.color1 = (0.2, 0.2, 0.2)

        initial_x = W / 2 + W * np.random.uniform(-0.3, 0.3)
        initial_y = H * 0.95
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((-ROCKET_WIDTH / 2, 0),
                                             (+ROCKET_WIDTH / 2, 0),
                                             (ROCKET_WIDTH / 2, +ROCKET_HEIGHT),
                                             (-ROCKET_WIDTH / 2, +ROCKET_HEIGHT))),
                density=1.0,
                friction=0.5,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0)
        )

        self.lander.color1 = rgb(230, 230, 230)

        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY, initial_y + ROCKET_WIDTH * 0.2),
                angle=(i * BASE_ANGLE),
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=((0, 0), (0, LEG_LENGTH / 25), (i * LEG_LENGTH, 0), (i * LEG_LENGTH, -LEG_LENGTH / 20),
                                  (i * LEG_LENGTH / 3, -LEG_LENGTH / 7))),
                    density=1,
                    restitution=0.0,
                    friction=0.2,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.25, 0.25, 0.25)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(i * LEG_AWAY, ROCKET_WIDTH * 0.2),
                localAnchorB=(0, 0),
                enableLimit=True,
                maxMotorTorque=2500.0,
                motorSpeed=-0.05 * i,
                enableMotor=True
            )
            djd = distanceJointDef(bodyA=self.lander,
                                   bodyB=leg,
                                   anchorA=(i * LEG_AWAY, ROCKET_HEIGHT / 8),
                                   anchorB=leg.fixtures[0].body.transform * (i * LEG_LENGTH, 0),
                                   collideConnected=False,
                                   frequencyHz=0.01,
                                   dampingRatio=0.9
                                   )
            if i == 1:
                rjd.lowerAngle = -SPRING_ANGLE
                rjd.upperAngle = 0
            else:
                rjd.lowerAngle = 0
                rjd.upperAngle = + SPRING_ANGLE
            leg.joint = self.world.CreateJoint(rjd)
            leg.joint2 = self.world.CreateJoint(djd)

            self.legs.append(leg)

        self.lander.linearVelocity = (
            -self.np_random.uniform(0, INITIAL_RANDOM) * START_SPEED * (initial_x - W / 2) / W,
            -START_SPEED)

        self.lander.angularVelocity = (1 + INITIAL_RANDOM) * np.random.uniform(-1, 1)

        self.drawlist = self.legs + [self.water] + [self.ship] + self.containers + [self.lander]

        if CONTINUOUS:
            return self.step([0, 0, 0])[0]
        else:
            return self.step(6)[0]

    def step(self, action):

        self.force_dir = 0

        if CONTINUOUS:
            np.clip(action, -1, 1)
            self.gimbal += action[0] * 0.15 / FPS
            self.throttle += action[1] * 0.5 / FPS
            if action[2] > 0.5:
                self.force_dir = 1
            elif action[2] < -0.5:
                self.force_dir = -1
        else:
            if action == 0:
                self.gimbal += 0.01
            elif action == 1:
                self.gimbal -= 0.01
            elif action == 2:
                self.throttle += 0.01
            elif action == 3:
                self.throttle -= 0.01
            elif action == 4:  # left
                self.force_dir = -1
            elif action == 5:  # right
                self.force_dir = 1

        self.gimbal = np.clip(self.gimbal, -GIMBAL_THRESHOLD, GIMBAL_THRESHOLD)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.power = 0 if self.throttle == 0.0 else MIN_THROTTLE + self.throttle * (1 - MIN_THROTTLE)

        # main engine force
        force_pos = (self.lander.position[0], self.lander.position[1])
        force = (-np.sin(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.power,
                 np.cos(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.power)
        self.lander.ApplyForce(force=force, point=force_pos, wake=False)

        # control thruster force
        force_pos_c = self.lander.position + THRUSTER_HEIGHT * np.array(
            (np.sin(self.lander.angle), np.cos(self.lander.angle)))
        force_c = (-self.force_dir * np.cos(self.lander.angle) * SIDE_ENGINE_POWER,
                   self.force_dir * np.sin(self.lander.angle) * SIDE_ENGINE_POWER)
        self.lander.ApplyLinearImpulse(impulse=force_c, point=force_pos_c, wake=False)

        self.world.Step(1.0 / FPS, 60, 60)

        pos = self.lander.position
        vel_l = np.array(self.lander.linearVelocity) / START_SPEED
        vel_a = self.lander.angularVelocity
        x_distance = (pos.x - W / 2) / W
        y_distance = (pos.y - self.shipheight) / (H - self.shipheight)

        angle = (self.lander.angle / np.pi) % 2
        if angle > 1:
            angle -= 2

        state = [
            2 * x_distance,
            2 * (y_distance - 0.5),
            angle,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            2 * (self.throttle - 0.5),
            (self.gimbal / GIMBAL_THRESHOLD)
        ]
        if VEL_STATE:
            state.extend([vel_l[0],
                          vel_l[1],
                          vel_a])

        # REWARD -------------------------------------------------------------------------------------------------------

        # state variables for reward
        distance = np.linalg.norm((3 * x_distance, y_distance))  # weight x position more
        speed = np.linalg.norm(vel_l)
        groundcontact = self.legs[0].ground_contact or self.legs[1].ground_contact
        brokenleg = (self.legs[0].joint.angle < 0 or self.legs[1].joint.angle > -0) and groundcontact
        outside = abs(pos.x - W / 2) > W / 2 or pos.y > H
        fuelcost = 0.1 * (0 * self.power + abs(self.force_dir)) / FPS
        landed = self.legs[0].ground_contact and self.legs[1].ground_contact and speed < 0.1
        done = False

        reward = -fuelcost

        if outside or brokenleg:
            self.game_over = True

        if self.game_over:
            done = True
        else:
            # reward shaping
            shaping = -0.5 * (distance + speed + abs(angle) ** 2)
            shaping += 0.1 * (self.legs[0].ground_contact + self.legs[1].ground_contact)
            if self.prev_shaping is not None:
                reward += shaping - self.prev_shaping
            self.prev_shaping = shaping

            if landed:
                self.landed_ticks += 1
            else:
                self.landed_ticks = 0
            if self.landed_ticks == FPS:
                reward = 1.0
                done = True

        if done:
            reward += max(-1, 0 - 2 * (speed + distance + abs(angle) + abs(vel_a)))
        elif not groundcontact:
            reward -= 0.25 / FPS

        reward = np.clip(reward, -1, 1)

        # REWARD -------------------------------------------------------------------------------------------------------

        self.stepnumber += 1

        state = (state - MEAN[:len(state)]) / VAR[:len(state)]

        return np.array(state), reward, done, {}

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering

        if self.viewer is None:

            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, W, 0, H)

            sky = rendering.FilledPolygon(((0, 0), (0, H), (W, H), (W, 0)))
            self.sky_color = rgb(126, 150, 233)
            sky.set_color(*self.sky_color)
            self.sky_color_half_transparent = np.array((np.array(self.sky_color) + rgb(255, 255, 255))) / 2
            self.viewer.add_geom(sky)

            self.rockettrans = rendering.Transform()

            engine = rendering.FilledPolygon(((0, 0),
                                              (ENGINE_WIDTH / 2, -ENGINE_HEIGHT),
                                              (-ENGINE_WIDTH / 2, -ENGINE_HEIGHT)))
            self.enginetrans = rendering.Transform()
            engine.add_attr(self.enginetrans)
            engine.add_attr(self.rockettrans)
            engine.set_color(.4, .4, .4)
            self.viewer.add_geom(engine)

            self.fire = rendering.FilledPolygon(((ENGINE_WIDTH * 0.4, 0), (-ENGINE_WIDTH * 0.4, 0),
                                                 (-ENGINE_WIDTH * 1.2, -ENGINE_HEIGHT * 5),
                                                 (0, -ENGINE_HEIGHT * 8), (ENGINE_WIDTH * 1.2, -ENGINE_HEIGHT * 5)))
            self.fire.set_color(*rgb(255, 230, 107))
            self.firescale = rendering.Transform(scale=(1, 1))
            self.firetrans = rendering.Transform(translation=(0, -ENGINE_HEIGHT))
            self.fire.add_attr(self.firescale)
            self.fire.add_attr(self.firetrans)
            self.fire.add_attr(self.enginetrans)
            self.fire.add_attr(self.rockettrans)

            smoke = rendering.FilledPolygon(((ROCKET_WIDTH / 2, THRUSTER_HEIGHT * 1),
                                             (ROCKET_WIDTH * 3, THRUSTER_HEIGHT * 1.03),
                                             (ROCKET_WIDTH * 4, THRUSTER_HEIGHT * 1),
                                             (ROCKET_WIDTH * 3, THRUSTER_HEIGHT * 0.97)))
            smoke.set_color(*self.sky_color_half_transparent)
            self.smokescale = rendering.Transform(scale=(1, 1))
            smoke.add_attr(self.smokescale)
            smoke.add_attr(self.rockettrans)
            self.viewer.add_geom(smoke)

            self.gridfins = []
            for i in (-1, 1):
                finpoly = (
                    (i * ROCKET_WIDTH * 1.1, THRUSTER_HEIGHT * 1.01),
                    (i * ROCKET_WIDTH * 0.4, THRUSTER_HEIGHT * 1.01),
                    (i * ROCKET_WIDTH * 0.4, THRUSTER_HEIGHT * 0.99),
                    (i * ROCKET_WIDTH * 1.1, THRUSTER_HEIGHT * 0.99)
                )
                gridfin = rendering.FilledPolygon(finpoly)
                gridfin.add_attr(self.rockettrans)
                gridfin.set_color(0.25, 0.25, 0.25)
                self.gridfins.append(gridfin)

        if self.stepnumber % round(FPS / 10) == 0 and self.power > 0:
            s = [MAX_SMOKE_LIFETIME * self.power,  # total lifetime
                 0,  # current lifetime
                 self.power * (1 + 0.2 * np.random.random()),  # size
                 np.array(self.lander.position)
                 + self.power * ROCKET_WIDTH * 10 * np.array((np.sin(self.lander.angle + self.gimbal),
                                                              -np.cos(self.lander.angle + self.gimbal)))
                 + self.power * 5 * (np.random.random(2) - 0.5)]  # position
            self.smoke.append(s)

        for s in self.smoke:
            s[1] += 1
            if s[1] > s[0]:
                self.smoke.remove(s)
                continue
            t = rendering.Transform(translation=(s[3][0], s[3][1] + H * s[1] / 2000))
            self.viewer.draw_circle(radius=0.05 * s[1] + s[2],
                                    color=self.sky_color + (1 - (2 * s[1] / s[0] - 1) ** 2) / 3 * (
                                            self.sky_color_half_transparent - self.sky_color)).add_attr(t)

        self.viewer.add_onetime(self.fire)
        for g in self.gridfins:
            self.viewer.add_onetime(g)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)

        for l in zip(self.legs, [-1, 1]):
            path = [self.lander.fixtures[0].body.transform * (l[1] * ROCKET_WIDTH / 2, ROCKET_HEIGHT / 8),
                    l[0].fixtures[0].body.transform * (l[1] * LEG_LENGTH * 0.8, 0)]
            self.viewer.draw_polyline(path, color=self.ship.color1, linewidth=1 if START_HEIGHT > 500 else 2)

        self.viewer.draw_polyline(((self.helipad_x2, self.terrainheigth + SHIP_HEIGHT),
                                   (self.helipad_x1, self.terrainheigth + SHIP_HEIGHT)),
                                  color=rgb(206, 206, 2),
                                  linewidth=1)

        self.rockettrans.set_translation(*self.lander.position)
        self.rockettrans.set_rotation(self.lander.angle)
        self.enginetrans.set_rotation(self.gimbal)
        self.firescale.set_scale(newx=1, newy=self.power * np.random.uniform(1, 1.3))
        self.smokescale.set_scale(newx=self.force_dir, newy=1)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255
