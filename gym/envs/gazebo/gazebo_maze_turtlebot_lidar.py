import gym
import rospy
import roslaunch
import time
import numpy as np

from gym.envs.gazebo import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty


class GazeboMazeTurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):

        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboMazeTurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)


        #THIS IS UNCLEAR
        self.action_space = spaces.Discrete(3) #F,L,R
        self.observation_space = spaces.Box(low=0, high=20) #laser values
        self.reward_range = (-np.inf, np.inf)

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

        #TEST
        if action == 1:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.5
            self.vel_pub.publish(vel_cmd)


        time.sleep(0.2)


        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
            resp_pause = pause.call()
        except rospy.ServiceException, e:
            print "/gazebo/pause_physics service call failed"



        #test params

        state = np.array([1, 2])
        reward = 1
        done = False

        return state, reward, done, {}
