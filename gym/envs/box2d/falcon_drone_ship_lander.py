import sys, math
import numpy as np
import pyglet

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import pygame
from pygame.locals import K_RIGHT, K_LEFT, KEYDOWN, KEYUP

import gym
from gym import spaces
from gym.utils import seeding


# Inspired by the existing OpenAI Gym LunarLander-v0 environment, as well as by Elon Musk's SpaceX crazy projects !
__author__ = "Victor Barbaros"
__credits__ = ["OpenAi Gym", "Oleg Klimov"]
__version__ = "0.0.1"
__maintainer__ = "Victor Barbaros"
__github_username__ = "vBarbaros"

FPS    = 60
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 100.0 # 13.0
SIDE_ENGINE_POWER  =  3.0 # 0.6

INITIAL_RANDOM = 500 #300.0   # Set 1500 to make game harder

FALCON_POLY =[
    (-14,+37), (-17,0), (-17,-30),
    (+17,-30), (+17,0), (+14,+37)
    ]

LEG_AWAY = 12 #20
LEG_DOWN = 30 # 18
LEG_W, LEG_H = 3, 10
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

DRONE_SHIP_H = 0.25
DRONE_SHIP_W = 2.5
SEA_LEVEL = 30/SCALE
GOING_LEFT = False
CONST_FORCE_DRONE_SHIP = 0.75
FREQUENCY_FACTOR = 250

VIEWPORT_W = 1400
VIEWPORT_H = 800


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.falcon_rocket==contact.fixtureA.body or self.env.falcon_rocket==contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

class FalconLander(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self):
        pygame.init()
        self._seed()
        self.viewer = None
        
        self.world = Box2D.b2World()
        self.sea_surface = None
        self.falcon_rocket = None
        self.floating_drone_ship = None
        self.particles = []

        self.prev_reward = None

        high = np.array([np.inf]*8)  # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-high, high)
        # Check the action space and define appropriately in the new heuristic funciton
        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,))
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self._reset()
    

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _destroy(self):
        if not self.sea_surface: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.sea_surface)
        self.sea_surface = None

        self.world.DestroyBody(self.floating_drone_ship)
        self.floating_drone_ship = None
        
        self.world.DestroyBody(self.falcon_rocket)
        self.falcon_rocket = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])


    def _reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE
        
        # create the sea surface into which the flacon might draw (with no collision)
        f = fixtureDef(
            shape=polygonShape(box=(W,SEA_LEVEL), radius=0.0), 
            density=0.5, 
            friction=0.03)

        self.sea_surface = self.world.CreateStaticBody(
            position=(0, SEA_LEVEL), angle=0,
            fixtures = f)

        # Create the floating drone ship
        f = fixtureDef(
            shape=polygonShape(box=(DRONE_SHIP_W,DRONE_SHIP_H), 
            radius=0.0), density=0.4, friction=0.05,
            categoryBits=0x0020,
            maskBits=0x001) #, userData=self.logo_img)

        self.floating_drone_ship = self.world.CreateDynamicBody(
            position=(DRONE_SHIP_W/SCALE, SEA_LEVEL), angle=0, linearDamping = 0.7, angularDamping = 0.3,
            fixtures = f)
        
        self.floating_drone_ship.color1 = (0.1,0.1,0.1)
        self.floating_drone_ship.color2 = (0.2,0.2,0.2)

        self.sea_surface.color1 = (0.5,0.4,0.9)
        self.sea_surface.color2 = (0.3,0.3,0.5)
    

        initial_y = VIEWPORT_H/SCALE
        #  create the falcon lander
        self.falcon_rocket = self.world.CreateDynamicBody(
            position = (VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in FALCON_POLY ]),
                density=5.0, # 5.0
                friction=0.1,
                categoryBits=0x001,
                maskBits=0x0020,  # collide only with floating_drone_ship
                restitution=0.0) #, userData=self.logo_img) # 0.99 bouncy
                )
        self.falcon_rocket.color1 = (0.7,0.7,0.7)
        self.falcon_rocket.color2 = (0.2,0.2,0.2)

        self.falcon_rocket.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)

        # create the legs of the falcon rocket
        self.legs = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x001,
                    maskBits=0x0020)
                )
            leg.ground_contact = False
            leg.color1 = (0.5,0.4,0.9)
            leg.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.falcon_rocket,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i==-1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        
        self.drawlist = [self.falcon_rocket] + self.legs + [self.floating_drone_ship] + [self.sea_surface] 

        return self._step(np.array([0,0]) if self.continuous else 0)[0]


    def _create_particle(self, mass, x, y, ttl):
    	# create those butifull particles that bubble off when the force is applied 
        p = self.world.CreateDynamicBody(
            position = (x,y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0,0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl<0):
            self.world.DestroyBody(self.particles.pop(0))

    
    def control_floating_platform(self):
        global GOING_LEFT
        fx = 1
        fy = 0
        p1 = self.floating_drone_ship.GetWorldPoint(localPoint=(3.25, 5.5))
        p2 = self.floating_drone_ship.GetWorldPoint(localPoint=(3.75, 7.5))
        width, fy = p1[0], p1[1]
        
        new_y_p1 = 2.5*math.cos(FREQUENCY_FACTOR)
        new_y_p2 = 2.5*math.sin(FREQUENCY_FACTOR)

        if not GOING_LEFT:
            self.floating_drone_ship.ApplyForce(force=(CONST_FORCE_DRONE_SHIP*fx,CONST_FORCE_DRONE_SHIP*new_y_p2), point=p2, wake=True)
            self.floating_drone_ship.ApplyForce(force=(CONST_FORCE_DRONE_SHIP/2*fx,CONST_FORCE_DRONE_SHIP/2*new_y_p1), point=p1, wake=True)
            if width > (VIEWPORT_W/SCALE*0.7):
                GOING_LEFT = True
        else:
            self.floating_drone_ship.ApplyForce(force=((-1)*CONST_FORCE_DRONE_SHIP*fx,-CONST_FORCE_DRONE_SHIP*new_y_p1), point=p1, wake=True)
            self.floating_drone_ship.ApplyForce(force=((-1)*CONST_FORCE_DRONE_SHIP/2*fx,-CONST_FORCE_DRONE_SHIP/2*new_y_p2), point=p2, wake=True)
            if width <= (VIEWPORT_W/SCALE*0.3):
                GOING_LEFT = False


    def _step(self, action):
        # act on the floating platform, it's autonomous
        self.control_floating_platform()

        # Engines
        tip  = (math.sin(self.falcon_rocket.angle), math.cos(self.falcon_rocket.angle))
        side = (-tip[1], tip[0]);
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action==2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power>=0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            orient_x =  tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
            orient_y = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
            impulse_pos = (self.falcon_rocket.position[0] + orient_x, self.falcon_rocket.position[1] + orient_y)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)    # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse(           ( orient_x*MAIN_ENGINE_POWER*m_power,  orient_y*MAIN_ENGINE_POWER*m_power), impulse_pos, True)
            self.falcon_rocket.ApplyLinearImpulse( (-orient_x*MAIN_ENGINE_POWER*m_power, -orient_y*MAIN_ENGINE_POWER*m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1,3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5,1.0)
                assert s_power>=0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            orient_x =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            orient_y = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (self.falcon_rocket.position[0] + orient_x - tip[0]*17/SCALE, self.falcon_rocket.position[1] + orient_y + tip[1]*SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(           ( orient_x*SIDE_ENGINE_POWER*s_power,  orient_y*SIDE_ENGINE_POWER*s_power), impulse_pos, True)
            self.falcon_rocket.ApplyLinearImpulse( (-orient_x*SIDE_ENGINE_POWER*s_power, -orient_y*SIDE_ENGINE_POWER*s_power), impulse_pos, True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.falcon_rocket.position
        vel = self.falcon_rocket.linearVelocity

        pos_floating_drone_ship = self.floating_drone_ship.position
        vel_floating_drone_ship = self.floating_drone_ship.linearVelocity

        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (DRONE_SHIP_H + LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.falcon_rocket.angle,
            20.0*self.falcon_rocket.angularVelocity,
            1.0 if (self.legs[0].ground_contact and self.legs[1].ground_contact) else 0.0,
            1.0 if (self.legs[0].ground_contact and self.legs[1].ground_contact) else 0.0,
            # add the (x and dx of the floating drone ship)
            (pos_floating_drone_ship.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            pos_floating_drone_ship.y / (VIEWPORT_H/SCALE/2),
            vel_floating_drone_ship.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel_floating_drone_ship.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.floating_drone_ship.angle
            ]

        assert len(state)==13

        reward = 0

        shaping = \
            - 100*np.sqrt( (state[0] - state[8])**2 + (state[1] - state[9])**2) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7] \
            - 100*abs(state[8]) - 100*abs(state[10]) - 100*abs(state[8]) 
        # in the last line above: higher values of x, dx and angle of drone ship => higher reward

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.10  # In the floating ship version, the penalty should be smaller
        reward -= s_power*0.01

        done = False

        DRONE_LEVEL = state[9]
        if self.game_over or state[1] < DRONE_LEVEL:
            done   = True
            reward = -150
        if not self.falcon_rocket.awake:
            done   = True
            reward = +150
        
        return np.array(state), reward, done, {}


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))
            obj.color2 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))

        self._clean_particles(False)

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class FalconLanderContinuous(FalconLander):
    continuous = True

def heuristic(env, s):
    # Heuristic for:
    # 1. Testing. 
    # 2. Demonstration rollout.

    #PID to get closer to the floating drone ship
    cross_err = math.sqrt( (s[0] - (s[8]))**2 + (s[1] - s[9])**2 )
    #print 'cross_error: ', cross_err
    angle_targ = (s[0]-s[8])*0.5 + s[2]*0.85         # angle should point towards center (s[0] is horizontal coordinate, s[2] horiz. speed)
    if angle_targ >  0.3: 
        angle_targ =  0.3  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.3: 
        angle_targ = -0.3
    
    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.55 - (s[5])*0.25
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))
    
    hover_targ = 0.25*(cross_err)  
    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ/s[1] - s[1])*0.6 - (s[3])*0.4
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))
    
    if s[6] or s[7]: # both legs have contact 
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array( [hover_todo*30 - 1, -angle_todo*30] )
        a = np.clip(a, -1, +1)

    return a



def key_control(env, s):
    action_done = False
    action = [0, 0]
    #print type(env.logo_img)
    #print env.logo_img
    # Action is two floats [main engine, left-right engines].
    # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
    # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            action_done = True
            break
        if event.key == K_RIGHT:
            action[0] = 0.0
            action[1] = 1.0
            action_done = True
                    
        elif event.key == K_LEFT:
            action[0] =   0.0
            action[1] =  -1.0
            action_done = True
                        
        elif event.type == KEYUP:
            action[0] = 1
            action[1] = 0
            action_done = True
                        
    return np.array(action)

if __name__=="__main__":
    #env = FalconLander()
    env = FalconLanderContinuous()
    s = env.reset()
    total_reward = 0
    steps = 0
    while True:
        a = heuristic(env, s)
        #a = key_control(env, s)
        s, r, done, info = env.step(a)
        env.render()
        total_reward += r
        if steps % 20 == 0 or done:
            print(["{:+0.2f}".format(x) for x in s])
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
