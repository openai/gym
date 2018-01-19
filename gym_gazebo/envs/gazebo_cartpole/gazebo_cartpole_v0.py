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

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class GazeboCartPolev0Env(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCartPole_v0.launch")

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self._pub = rospy.Publisher('/cart_pole_controller/command', JointTrajectory)
        # self._sub = rospy.Subscriber('/joint_states', JointState, self.observation_callback)

        # Gazebo specific services to start/stop its behavior and
        # facilitate the overall RL environment
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # Seed the environment
        self._seed()
        self.steps_beyond_done = None

        self.last_pose = 0

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg =  message

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/unpause_physics service call failed")

        action_msg = JointTrajectory()
        action_msg.header.stamp = rospy.Time.now() # Note you need to call rospy.init_node() before this will work
        action_msg.joint_names = ["slider_to_cart"]

        self.last_pose = action;

        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        # target.positions  = [self.last_pose]

        target.time_from_start.secs = 0
        target.time_from_start.nsecs = 1000000000

        target.velocities = [1]
        # target.effort = [float('nan')]

        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]

        self._pub.publish(action_msg)


        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=5)
            except:
                pass

        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")

        state = [data.position[1], 0, data.position[0], 0]

        x, x_dot, theta, theta_dot = state

        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

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

        return state, reward, done, {}

    def _reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # # Unpause simulation to make observation
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/unpause_physics service call failed")

        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=5)
            except:
                pass

        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")

        state = [data.position[1], 0, data.position[0], 0]

        self.steps_beyond_done = None

        return state
