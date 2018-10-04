import gym
import rospy
import roslaunch
import time
import numpy as np
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gym.utils import seeding
import copy
import math
import os

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState

class GazeboCartPolev0Env(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCartPole_v0.launch")

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 15

        self._pub = rospy.Publisher('/cart_pole_controller/command', Float64, queue_size=1)
        # self._sub = rospy.Subscriber('/joint_states', JointState, self.observation_callback)
        # self._sub = rospy.Subscriber('/cart_pole/joint_states', JointState, self.observation_callback)

        # Gazebo specific services to start/stop its behavior and
        # facilitate the overall RL environment
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.set_link = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)

        rospy.wait_for_service('/gazebo/set_link_state')

        # Seed the environment
        self._seed()
        self.steps_beyond_done = None

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.current_vel = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)
        rospy.Subscriber("/cart_pole/joint_states", JointState, self.callback)

    def callback(self, data):
        self.data = data

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        if self.data==None:
            while self.data is None:
                try:
                    self.data = rospy.wait_for_message('/cart_pole/joint_states', JointState, timeout=5)
                except:
                    pass
        angle = math.atan(math.tan(self.data.position[0]))

        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/unpause_physics service call failed")

        if action > 0.5:
            self.current_vel = self.current_vel + 1
        else:
            self.current_vel = self.current_vel - 1

        # print(os.environ["ROS_MASTER_URI"] )
        # print("action", action)

        action_msg = Float64()
        action_msg.data = self.current_vel
        # self.set_ros_master_uri();
        self._pub.publish(action_msg)

        # data = None
        # while data is None:
        #     try:
        #         data = rospy.wait_for_message('/cart_pole/joint_states', JointState, timeout=5)
        #     except:
        #         pass

        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")

        state = [self.data.position[1], self.data.velocity[1], angle, self.data.velocity[0]]
        # state = [self.data.position[1], 0, self.data.position[0], 0]

        x, x_dot, theta, theta_dot = state

        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        # if done:
        #     print("x: ", x,  " theta: ", theta*180/3.1416 )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        # # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")


        return state, reward, done, {}

    def reset(self):
        # # Resets the state of the environment and returns an initial observation.
        # rospy.wait_for_service('/gazebo/reset_simulation')
        # try:
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/reset_simulation service call failed")

        # # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        rospy.wait_for_service('/gazebo/set_link_state')
        self.set_link(LinkState(link_name='pole'))
        self.set_link(LinkState(link_name='cart'))

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        #read laser data
        self.data = None
        while self.data is None:
            try:
                self.data = rospy.wait_for_message('/cart_pole/joint_states', JointState, timeout=5)
            except:
                pass
        angle = math.atan(math.tan(self.data.position[0]))
        state = [self.data.position[1], self.data.velocity[1], angle, self.data.velocity[0]]
        # state = [self.data.position[1], 0, self.data.position[0], 0]

        self.steps_beyond_done = None
        self.current_vel = 0

        return state
