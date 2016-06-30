import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering


class HoodleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_start_v = 0  # min start velocity
        self.max_start_v = 50
        self.goal_distance = 300  # distance of the hoodle move to goal_distance position
        self.goal_region = 40
        self.start_pos = 100  # start position
        self.min_distance = 0
        self.max_distance = 450  # 416.667 when max_start_v = 50
        self.hoodle_stop_distance = -1
        self.hoodle_stop_flag_color = (0, 0, 0)

        # state = (enable_set_start_v, goal_distance, cur_distance, velocity_start)
        observation_low = np.array([0, self.goal_distance, self.min_distance,
                                    self.min_start_v])
        observation_high = np.array([1, self.goal_distance, self.max_distance, self.max_start_v])
        self.observation_space = spaces.Box(observation_low, observation_high)

        self.action_space = spaces.Discrete(self.max_start_v)

        self.viewer = None

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_state(self):
        self.state = np.array([self.enable_set_start_v, self.goal_distance,
                               self.min_distance, self.start_v])
        return self.state

    def _step(self, action):
        self.step_cnt += 1
        seconds = 10
        interval_t = 1 / 30.0 * seconds
        start_move_cnt = 10
        wait_reset_cnt = 10

        u = 0.3  # friction coefficient of the wood
        G = 10  #
        a = u * G
        t = (self.step_cnt - start_move_cnt) * interval_t
        if t >= 0:
            self.enable_set_start_v = 0
        else:
            t = 0

        if self.enable_set_start_v > 0.5:
            self.enable_set_start_v = 0
            self.start_v = action
            self.stop_distance = self.start_v ** 2 / (2 * a)
            self.max_t = self.start_v / a

        if self.arrival_max_distance is not True:
            self.distance = (2 * self.start_v - a * t) / 2 * t

        reward = 0
        done = False
        if self.arrival_max_distance is not True and t >= self.max_t:
            self.arrival_max_distance = True
            self.distance = self.stop_distance
            self.stop_step_cnt = self.step_cnt
        if self.arrival_max_distance and self.step_cnt > self.stop_step_cnt + wait_reset_cnt:
            done = True
            self.hoodle_stop_distance = self.distance
            self.hoodle_stop_flag_color = (0, 0, 0)
            if self.distance >= self.goal_distance - self.goal_region / 2 \
                    and self.distance <= self.goal_distance + self.goal_region / 2:
                self.get_goal_distance = True
                reward = 1
                self.hoodle_stop_flag_color = (1.0, 0, 0)

        return self._get_state(), reward, done, {}

    def _reset(self):
        self.distance = 0
        self.arrival_max_distance = False
        self.max_t = 10000  # can't reach 10000 when call step, the unit of the value is second
        self.stop_step_cnt = 10000
        self.step_cnt = 0  # step count
        self.get_goal_distance = False
        self.stop_distance = self.max_distance
        self.start_v = 0  # start velocity
        self.enable_set_start_v = 1
        return self._get_state()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        hoodle_radius = 8
        line_y = 100
        start_v_flag_height = 5
        start_v_flag_shap_len = 5
        start_v_flag_x = self.start_pos - self.max_start_v - hoodle_radius - start_v_flag_shap_len
        start_v_flag_y = line_y + hoodle_radius - start_v_flag_height / 2
        goal_position = self.start_pos + self.goal_distance

        if self.viewer is None:
            screen_width = 600
            screen_height = 200
            clearance = 10

            self.viewer = rendering.Viewer(screen_width, screen_height)

            line = rendering.Line((0, line_y), (screen_width, line_y))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)

            l, r, t, b = goal_position - self.goal_region / 2, goal_position + self.goal_region / 2, line_y - 5, line_y
            target_rect = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            target_rect.set_color(1.0, 0., 0.)
            self.viewer.add_geom(target_rect)

            self.stop_hoodle_flag = rendering.make_circle(hoodle_radius, filled=False)
            self.stop_hoodle_flag.set_color(1, 1, .1)
            self.stop_hoodle_flag.add_attr(rendering.Transform(translation=(0, clearance)))
            self.stop_hoodle_flag_trans = rendering.Transform()
            self.stop_hoodle_flag.add_attr(self.stop_hoodle_flag_trans)
            self.viewer.add_geom(self.stop_hoodle_flag)

            self.hoodle = rendering.make_circle(hoodle_radius)
            self.hoodle.set_color(0, 1.0, .0)
            self.hoodle.add_attr(rendering.Transform(translation=(0, clearance)))
            self.hoodle_trans = rendering.Transform()
            self.hoodle.add_attr(self.hoodle_trans)
            self.viewer.add_geom(self.hoodle)

            start_flag = rendering.FilledPolygon([(self.start_pos, line_y), (self.start_pos - 10, line_y - 10),
                                                  (self.start_pos + 10, line_y - 10)])
            start_flag.set_color(.0, .0, .0)
            self.viewer.add_geom(start_flag)

            max_start_v_flag = rendering.FilledPolygon(
                [(start_v_flag_x, start_v_flag_y),
                 (start_v_flag_x + self.max_start_v, start_v_flag_y),
                 (start_v_flag_x + self.max_start_v + start_v_flag_shap_len,
                  start_v_flag_y + start_v_flag_height / 2),
                 (start_v_flag_x + self.max_start_v, start_v_flag_y + start_v_flag_height),
                 (start_v_flag_x, start_v_flag_y + start_v_flag_height)])
            max_start_v_flag.set_color(1.0, 1.0, 1.0)
            self.viewer.add_geom(max_start_v_flag)

            max_start_v_flag_line = rendering.PolyLine(
                [(start_v_flag_x, start_v_flag_y),
                 (start_v_flag_x + self.max_start_v, start_v_flag_y),
                 (start_v_flag_x + self.max_start_v + start_v_flag_shap_len,
                  start_v_flag_y + start_v_flag_height / 2 + 1),
                 (start_v_flag_x + self.max_start_v, start_v_flag_y + start_v_flag_height),
                 (start_v_flag_x, start_v_flag_y + start_v_flag_height)], False)
            max_start_v_flag_line.set_color(.0, .0, .0)
            self.viewer.add_geom(max_start_v_flag_line)

            self.start_v_flag = rendering.FilledPolygon(
                [(start_v_flag_x, start_v_flag_y + 20), (start_v_flag_x + self.start_v, start_v_flag_y),
                 (start_v_flag_x + self.start_v + 5, start_v_flag_y + start_v_flag_height / 2),
                 (start_v_flag_x + self.start_v, start_v_flag_y + start_v_flag_height),
                 (start_v_flag_x, start_v_flag_y + start_v_flag_height)])
            self.start_v_flag.set_color(.0, .0, .0)
            self.viewer.add_geom(self.start_v_flag)

        if self.hoodle_stop_distance != -1:  #
            self.stop_hoodle_flag.set_color(self.hoodle_stop_flag_color[0],
                                            self.hoodle_stop_flag_color[1],
                                            self.hoodle_stop_flag_color[2])
            if self.get_goal_distance:
                self.stop_hoodle_flag_trans.set_translation(self.start_pos + self.hoodle_stop_distance,
                                                            line_y)
            else:
                self.stop_hoodle_flag_trans.set_translation(self.start_pos + self.hoodle_stop_distance,
                                                            line_y)

        self.start_v_flag.v = [(start_v_flag_x, start_v_flag_y),
                               (start_v_flag_x + self.start_v, start_v_flag_y),
                               (start_v_flag_x + self.start_v + start_v_flag_shap_len,
                                start_v_flag_y + start_v_flag_height / 2 + 1),
                               (start_v_flag_x + self.start_v, start_v_flag_y + start_v_flag_height),
                               (start_v_flag_x, start_v_flag_y + start_v_flag_height)]
        self.hoodle_trans.set_translation(self.start_pos + self.distance, line_y)

        self.viewer.render()
        if mode == 'rgb_array':
            return self.viewer.get_array()
        elif mode == 'human':
            pass
