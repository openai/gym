import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym.envs.gazebo import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

class GazeboMazeTurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):

        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboMazeTurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)


        #THIS IS UNCLEAR
        #self.action_space = spaces.Discrete(3) #F,L,R
        #self.observation_space = spaces.Box(low=0, high=20) #laser values
        #self.reward_range = (-np.inf, np.inf)

    def _step(self, action):

        # TODO
        # Perform a step in gazebo. E.g. move the robot
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            pause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            resp_pause = pause.call()
        except rospy.ServiceException, e:
            print "/gazebo/unpause_physics service call failed"

        #self.move()
        #step delimiter (time or position change)
        #time.sleep(0.05) 


        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.angular.z = -0.5
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.angular.z = 0.5
            self.vel_pub.publish(vel_cmd)

        time.sleep(0.2)


        #read laser data
        data = rospy.wait_for_message('/scan', LaserScan, timeout=5)

        #simplify ranges - discretize

        discretized_ranges = []
        discretized_ranges_amount = 10

        mod = (len(data.ranges) / discretized_ranges_amount)
        for i, item in enumerate(data.ranges):
            if (i%mod==0) and (i!=0):
                discretized_ranges.append(int(data.ranges[i]))


        state = discretized_ranges 


        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
            resp_pause = pause.call()
        except rospy.ServiceException, e:
            print "/gazebo/pause_physics service call failed"



        #test params



        reward = 1
        done = False

        return state, reward, done, {}
