"""
PuckWorld Environment for OpenAI gym

The data used in this model comes from:
http://cs.stanford.edu/people/karpathy/reinforcejs/puckworld.html


Author: Qiang Ye
Date: July 17, 2017
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

RAD2DEG = 57.29577951308232     # 弧度与角度换算关系1弧度=57.29..角度

class PuckWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
        }

    def __init__(self):
        self.width = 600            # 场景宽度 screen width
        self.height = 600           # 场景长度
        self.l_unit = 1.0           # 场景长度单位 pysical world width
        self.v_unit = 1.0           # 速度单位 velocity 
        self.max_speed = 0.025      # max agent velocity along a axis
        
        self.re_pos_interval = 30   # 目标重置距离时间
        self.accel = 0.002          # agent 加速度
        self.rad = 0.05             # agent 半径,目标半径
        self.target_rad = 0.01      # target radius.
        self.goal_dis = self.rad    # 目标接近距离 expected goal distance
        self.t = 0                  # puck world clock
        self.update_time = 100      # time for target randomize its position
        # 作为观察空间每一个特征值的下限
        self.low = np.array([0,        # agent position x
                            0,
                            -self.max_speed,    # agent velocity
                            -self.max_speed,    
                            0,         # target position x
                            0,
                            ])   
        self.high = np.array([self.l_unit,                 
                            self.l_unit,
                            self.max_speed,    
                            self.max_speed,    
                            self.l_unit,    
                            self.l_unit,
                            ])   
        self.reward = 0         # for rendering
        self.action = None      # for rendering
        self.viewer = None
        # 0,1,2,3,4 represent left, right, up, down, -, five moves.
        self.action_space = spaces.Discrete(5)  
        # 观察空间由low和high决定
        self.observation_space = spaces.Box(self.low, self.high)    

        self._seed()    # 产生一个随机数种子
        self.reset()

    def _seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)  
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        
        self.action = action    # action for rendering
        ppx,ppy,pvx,pvy,tx,ty = self.state # 获取agent位置，速度，目标位置
        ppx, ppy = ppx+pvx, ppy+pvy         # update agent position
        pvx, pvy = pvx*0.95, pvy*0.95       # natural velocity loss

        if action == 0: pvx -= self.accel   # left
        if action == 1: pvx += self.accel   # right
        if action == 2: pvy += self.accel   # up
        if action == 3: pvy -= self.accel   # down
        if action == 4: pass                # no move

        if ppx < self.rad:              # encounter left bound
            pvx *= -0.5
            ppx = self.rad
        if ppx > 1 - self.rad:          # right bound
            pvx *= -0.5
            ppx = 1 - self.rad
        if ppy < self.rad:              # bottom bound
            pvy *= -0.5
            ppy = self.rad
        if ppy > 1 - self.rad:          # right bound
            pvy *= -0.5
            ppy = 1 - self.rad
        
        self.t += 1
        if self.t % self.update_time == 0:  # update target position
            tx = self._random_pos();        # randomly
            ty = self._random_pos();

        dx, dy = ppx - tx, ppy - ty         # calculate distance from
        dis = self._compute_dis(dx, dy)     # agent to target

        self.reward = self.goal_dis - dis   # give an reward

        done = bool(dis <= self.goal_dis)   
        
        self.state = (ppx, ppy, pvx, pvy, tx, ty)
        return np.array(self.state), self.reward, done, {}

    def _random_pos(self):
        return self.np_random.uniform(low = 0, high = self.l_unit)

    def _compute_dis(self, dx, dy):
        return math.sqrt(math.pow(dx,2) + math.pow(dy,2))

    def _reset(self):
        self.state = np.array([ self._random_pos(),
                                self._random_pos(),
                                0,
                                0,
                                self._random_pos(),
                                self._random_pos()
                               ])
        return self.state   # np.array(self.state)


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        scale = self.width/self.l_unit      # 计算两者映射关系
        rad = self.rad * scale              # agent radius, also positive reward area.
        t_rad = self.target_rad * scale     # target radius

        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # 在Viewer里绘制一个几何图像的步骤如下：
            # 1. 建立该对象需要的数据本身
            # 2. 使用rendering提供的方法返回一个geom对象
            # 3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个性化的方法
            #    来设置属性，具体请参考继承自这些Geom的对象），这其中有一个重要的属性就是
            #    变换属性，该属性负责对对象在屏幕中的位置、渲染、缩放进行渲染。如果某对象
            #    在呈现时可能发生上述变化，则应建立关于该对象的变换属性。该属性是一个
            #    Transform对象，而一个Transform对象，包括translate、rotate和scale
            #    三个属性，每个属性都由以np.array对象描述的矩阵决定。
            # 4. 将新建立的geom对象添加至viewer的绘制对象列表里，如果在屏幕上只出现一次，
            #    将其加入到add_onegeom(）列表中，如果需要多次渲染，则将其加入add_geom()
            # 5. 在渲染整个viewer之前，对有需要的geom的参数进行修改，修改主要基于该对象
            #    的Transform对象
            # 6. 调用Viewer的render()方法进行绘制

            target = rendering.make_circle(t_rad, 30, True)
            target.set_color(0.1, 0.9, 0.1)
            self.viewer.add_geom(target)
            target_circle = rendering.make_circle(t_rad, 30, False)
            target_circle.set_color(0, 0, 0)
            self.viewer.add_geom(target_circle)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            target_circle.add_attr(self.target_trans)

            self.agent = rendering.make_circle(rad, 30, True)
            self.agent.set_color(0, 1, 0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

            agent_circle = rendering.make_circle(rad, 30, False)
            agent_circle.set_color(0, 0, 0)
            agent_circle.add_attr(self.agent_trans)
            self.viewer.add_geom(agent_circle)
            # uncomment following to show a complete arrow rather than just an arrow head.
            # start_p = (0, 0)
            # end_p = (0.7 * rad, 0)
            # self.line = rendering.Line(start_p, end_p)
            # self.line.linewidth = rad / 10
            self.line_trans = rendering.Transform()
            # self.line.add_attr(self.line_trans)
            # self.viewer.add_geom(self.line)
            self.arrow = rendering.FilledPolygon([
                (0.7*rad,0.15*rad),
                (rad,0),
                (0.7*rad,-0.15*rad)
                ])
            self.arrow.set_color(0,0,0)
            self.arrow.add_attr(self.line_trans)
            self.viewer.add_geom(self.arrow)
            
        ppx,ppy,_,_,tx,ty = self.state
        self.target_trans.set_translation(tx*scale, ty*scale)
        self.agent_trans.set_translation(ppx*scale, ppy*scale)
        # 按距离给Agent着色, color agent accoring to its distance from target
        vv, ms = self.reward + 0.3, 1
        r, g, b, = 0, 1, 0
        if vv >= 0:
            r, g, b = 1 - ms*vv, 1, 1 - ms*vv
        else:
            r, g, b = 1, 1 + ms*vv, 1 + ms*vv 
        self.agent.set_color(r, g, b)
        
        a = self.action
        if a in [0,1,2,3]:
            #根据action绘制箭头
            degree = 0
            if a == 0: degree = 180
            elif a == 1: degree = 0
            elif a == 2: degree = 90
            else: degree = 270
            self.line_trans.set_translation(ppx*scale, ppy*scale)
            self.line_trans.set_rotation(degree/RAD2DEG)
            # self.line.set_color(0,0,0)
            self.arrow.set_color(0,0,0)
        else:           
            # self.line.set_color(r,g,b)
            self.arrow.set_color(r,g,b)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')


if __name__ =="__main__":
    env = PuckWorldEnv()
    print("hello")
    env.reset()
    for _ in range(10000):
        env.render()
        env.step(env.action_space.sample())

    print("env closed")
