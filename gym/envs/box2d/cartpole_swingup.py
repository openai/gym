import numpy as np
import gym
import math

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef)

from gym import spaces

FPS   = 50
SCALE = 120.0  # affects how fast-paced the game is
FORCE = 1.9    # 1.6 for underpowered setting, will need 3-4 swings to go up, longer episode

CART_WIDTH  = 25 / SCALE
CART_HEIGHT = 8  / SCALE
POLE_LENGTH = 60 / SCALE
POLE_WIDTH  = 2  / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

INITIAL_RANDOM = 10.0

class CartPoleSwingUp(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        self.viewer = None

        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf]) # useful range is -1 .. +1
        self.action_space = spaces.Discrete(3)  # nop, left, right
        self.observation_space = spaces.Box(-high, high)

        self.world = Box2D.b2World()
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.cart = None
        self.pole = None

        self.floor_body = self.world.CreateStaticBody(
            position = (VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/4 - CART_HEIGHT),
            fixtures = fixtureDef(
                shape=polygonShape(box=(VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/4)),
                friction=0.1
                )
            )
        self.floor_body.color1 = (0.6,0.9,0.6)
        self.floor_body.color2 = (0.6,0.9,0.6)

        self.prev_estimate = None
        self._reset()

    def _destroy(self):
        if not self.cart: return
        self.world.DestroyBody(self.cart)
        self.cart = None
        self.world.DestroyBody(self.pole)
        self.pole = None
        self.joint = None  # joint itself destroyed with bodies

    def _reset(self):
        self._destroy()

        self.cart = self.world.CreateDynamicBody(
            position = (VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2 + CART_HEIGHT/2),
            #angle=1.1,
            fixtures = fixtureDef(
                shape=polygonShape(box=(CART_WIDTH,CART_HEIGHT)),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.3) # 0.99 bouncy
                )
        self.cart.color1 = (0.5,0.4,0.9)
        self.cart.color2 = (0.3,0.3,0.5)
        self.cart.ApplyForceToCenter( (np.random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        self.pole = self.world.CreateDynamicBody(
            position = (VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2 + CART_HEIGHT/2 - POLE_LENGTH),
            angle = (0.0),
            fixtures = [
                fixtureDef(
                    shape=polygonShape(box=(POLE_WIDTH,POLE_LENGTH)),
                    density=1.0,
                    categoryBits=0x0020,
                    maskBits=0x000), # don't collide at all
                fixtureDef(
                    shape=circleShape(radius=POLE_WIDTH*2, pos=(0,-POLE_LENGTH)),
                    density=1.0,
                    categoryBits=0x0020,
                    maskBits=0x000)],
                )
        self.pole.color1 = (1.0,0,0.0)
        self.pole.color2 = (0.6,0,0.0)

        rjd = revoluteJointDef(
            bodyA=self.cart,
            bodyB=self.pole,
            localAnchorA=(0, 0),
            localAnchorB=(0, POLE_LENGTH)
            )
        self.joint = self.world.CreateJoint(rjd)

        self.drawlist = [self.floor_body, self.cart, self.pole]

        return self._step(0)[0]

    def _step(self, action):
        assert action==0 or action==1 or action==2, "%r (%s) invalid " % (action,type(action))

        if action != 0:
            self.cart.ApplyForceToCenter((-FORCE if action==1 else +FORCE, 0), True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.cart.position
        vel = self.cart.linearVelocity
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            math.sin( self.pole.angle ),
            math.cos( self.pole.angle ),
            0.2*self.pole.angularVelocity
            ]

        estimate  = -state[3]        # state[3] is -1 when the pole is up, so it's +1.0 in up position
        estimate -= abs(state[0])    # reduced by offset from center

        potential = False   # potential is easier to train
        if potential:
            # total reward received will be 2.0: from -1.0 (downward in center) to +1.0 (upward in center)
            reward = 0
            if self.prev_estimate is not None:
                reward = estimate - self.prev_estimate
            self.prev_estimate = estimate
        else:
            reward = estimate

        done = abs(state[0]) >= 1.0
        reward = estimate
        if done: reward = -1.0
        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        self.viewer.render()
        if mode == 'rgb_array':
            return self.viewer.get_array()
        elif mode is 'human':
            pass
        else:
            return super(CartPoleSwingUp, self).render(mode=mode)

