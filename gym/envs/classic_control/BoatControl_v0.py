import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.classic_control import rendering


class BoatControl(gym.Env):
    def __init__(self, t=0., dt=.1):
        self.dt = dt
        self.t = t
        high = np.array([np.deg2rad(10), 0.2])
        self.high = high
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.l = 0.252
        self.r = 0.018
        self.B = 0.987
        self.v = 0.5   # change
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.viewer = None
        self.theta = []
        self.seed()

    def f(self, t, y, z):
        return z

    def g(self, t, y, z):
        I_x = 184.56
        g_ = 9.81
        rho = 1025
        D = 1.679 * rho
        h_ = 0.1
        nu = 0.04
        Nu = nu * np.sqrt(I_x * D * h_)
        L = 6
        ol = np.random.uniform(0.75, 1.75)
        lamda = ol * L   # change
        k = 2 * np.pi / lamda
        H = 0.17 * lamda ** 0.75
        a = 0.5 * H
        fi = np.pi / 4   # change
        T = 0.8 * np.sqrt(lamda)
        W_0 = 2 * np.pi / T
        W_e = W_0 * (1 - self.v * W_0 * np.cos(fi) / g_)
        return (-D * h_ * k * a * np.sin(W_e * t) - 2 * Nu * z - D * h_ * y) / I_x

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def runge_kutta_z(self, t, y, z):
        k1 = self.f(t, y, z)
        l1 = self.g(t, y, z)
        k2 = self.f(t, y, z) + 0.5 * self.dt * l1
        l2 = self.g(t + 0.5 * self.dt, y + 0.5 * self.dt * k1, z + 0.5 * self.dt * l1)
        k3 = self.f(t, y, z) + 0.5 * self.dt * l2
        l3 = self.g(t + 0.5 * self.dt, y + 0.5 * self.dt * k2, z + 0.5 * self.dt * l2)
        l4 = self.g(t + 0.5 * self.dt, y + self.dt * k3, z + self.dt * l3)
        z = z + self.dt * (l1 + 2 * l2 + 2 * l3 + l4) / 6.
        return z

    def step(self, action):
        th, thdot = self.state
        rho = 1025
        I_x = 184.56
        Kc = 2 * np.pi * rho * action * self.r * self.r * self.v * self.l * (self.B + self.l)
        newthdot = self.runge_kutta_z(self.t, th, thdot) + Kc * 1000 / I_x
        newth = th + newthdot * self.dt
        self.t += self.dt
        self.state = np.array([newth, newthdot])
        done = False
        reward = -10 * np.abs(newth)
        if np.abs(newth) > np.deg2rad(35):
            reward = -100.
            done = True
        elif newth <= 1.:
            reward += 2
        elif newth > 0. and newthdot < 0.:
            reward += 0.5
        elif newth < 0. and newthdot > 0.:
            reward += 0.5
        elif newth > 0. and newthdot > 0.:
            reward += -0.5
        elif newth < 0. and newthdot < 0.:
            reward += -0.5
        self.theta.append(newth)
        if len(self.theta) == 500:
            plt.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
            plt.rcParams['axes.unicode_minus'] = False
            plt.plot(np.rad2deg(np.abs(self.theta)))
            plt.xlabel('Time(s)')
            plt.ylabel('Roll angel(°)')
            plt.savefig('boat_ddpg_theta1.png')
            plt.close()
            np.save('boat_ddpg_theta1.npy', self.theta)
            self.theta = []
        self.newth = newth
        reward = np.squeeze(reward)
        self.state = np.squeeze(self.state)
        return self.state, reward, done, {}

    def reset(self):
        high = np.array([np.deg2rad(2), 0.1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.state = np.array([self.state[0], self.state[1]])
        self.a = 10.
        return self.state

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500
        self.min_position = -1.0
        self.max_position = 1.0
        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-20, 20, -20, 20)
            xs = np.linspace(100 * self.min_position, 100 * self.max_position, 20000)
            ys = 0.6 * np.sin(0.5 * xs)
            xys = list(zip(xs, ys))
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.tracktrans = rendering.Transform()
            self.track.add_attr(self.tracktrans)
            self.viewer.add_geom(self.track)
            boatwidth = 3
            boatheight = 2
            l, r, t, b = -boatwidth / 2, boatwidth / 2, boatheight, 0
            self.boat = rendering.FilledPolygon([(l, b), (l - 1, t), (r + 1, t), (r, b)])
            self.boattrans = rendering.Transform()
            self.boat.add_attr(self.boattrans)
            self.viewer.add_geom(self.boat)
        self.boattrans.set_rotation(self.newth)
        return self.viewer.render(return_rgb_array=True)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
