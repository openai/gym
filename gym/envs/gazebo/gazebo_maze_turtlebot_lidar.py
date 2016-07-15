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

    '''def _spawn_robot(self):

        # TODO
        # Spawn robot
        # Optional
        # Another option is to spawn the robot in the __init__ method
        pass'''
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


    '''
    def _reset(self):

        # TODO
        # Reset world/simulation
        pass
    def _render(self, episodes):

        # Open GUI (if it's not allready opened?)
        # episodes = number of episodes that GUI is going to be opened. Another option is to use _close to close the gui
        super(GazeboMazeTurtlebotLidarEnv, self)._render()

    def _close(self):

        # TODO
        # From OpenAI API: Perform any necessary cleanup
        pass
    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass
    def _seed(self):
        
        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)  
        pass
'''