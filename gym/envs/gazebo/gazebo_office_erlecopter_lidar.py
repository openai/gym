import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym.envs.gazebo import gazebo_env
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class GazeboOfficeErleCopterLidarEnv(gazebo_env.GazeboEnv):

    def _takeoff(self, altitude):
        # Set throttle at 1500
        msg = OverrideRCIn()
        msg.channels[0] = 1500
        msg.channels[1] = 1500
        msg.channels[2] = 1500
        msg.channels[3] = 0
        msg.channels[4] = 0
        msg.channels[5] = 0
        msg.channels[6] = 0
        msg.channels[7] = 0
        self.pub.publish(msg)

        # Set GUIDED mode
        rospy.wait_for_service('mavros/set_mode')
        try:
            self.mode_proxy(0,'GUIDED')
        except rospy.ServiceException, e:
            print ("mavros/set_mode service call failed: %s"%e)

        # Wait 2 seconds
        time.sleep(2)

        # Arm throttle
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            self.arm_proxy(True)
        except rospy.ServiceException, e:
            print ("mavros/set_mode service call failed: %s"%e)

        # Takeoff
        rospy.wait_for_service('mavros/cmd/takeoff')
        try:
            self.takeoff_proxy(0, 0, 0, 0, altitude) # 1m altitude
        except rospy.ServiceException, e:
            print ("mavros/cmd/takeoff service call failed: %s"%e)

        # Wait 3 seconds
        time.sleep(3)

        # Set ALT_HOLD mode
        rospy.wait_for_service('mavros/set_mode')
        try:
            self.mode_proxy(0,'ALT_HOLD')
        except rospy.ServiceException, e:
            print ("mavros/set_mode service call failed: %s"%e)

    def __init__(self):

        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2ErleCopterLidar-v0.launch")    

        self.pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=10)

        self.action_space = spaces.Discrete(3) #F,L,R
        #self.observation_space = spaces.Box(low=0, high=20) #laser values
        self.reward_range = (-np.inf, np.inf)

        self.gazebo_step_size = long(200)


        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.mode_proxy = rospy.ServiceProxy('mavros/set_mode', SetMode)

        self.arm_proxy = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
        
        self.takeoff_proxy = rospy.ServiceProxy('mavros/cmd/takeoff', CommandTOL)

        countdown = 10
        while countdown > 0:
            print ("Taking off in in %ds"%countdown)
            countdown-=1
            time.sleep(1)

        self._takeoff(1)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _laser_state(self, action):
        

        return discretized_ranges, done

    def _step(self, action):

        msg = OverrideRCIn()

        if action == 0: #FORWARD
            msg.channels[0] = 1500 # Roll
            msg.channels[1] = 1400 # Pitch
        elif action == 1: #LEFT
            msg.channels[0] = 1400 # Roll
            msg.channels[1] = 1500 # Pitch
        elif action == 2: #RIGHT
            msg.channels[0] = 1600 # Roll
            msg.channels[1] = 1500 # Pitch

        msg.channels[2] = 1500  # Throttle
        msg.channels[3] = 0     # Yaw
        msg.channels[4] = 0
        msg.channels[5] = 0
        msg.channels[6] = 0
        msg.channels[7] = 0

        self.pub.publish(msg)
    
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        #simplify ranges - discretize
        discretized_ranges = []
        discretized_ranges_amount = 5
        min_range = 0.2 #collision

        done = False

        mod = (len(data.ranges) / discretized_ranges_amount)
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf'):
                    discretized_ranges.append(int(data.range_max))
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
                    #discretized_ranges.append(round(data.ranges[i] * 2) / 2)
            if (min_range > data.ranges[i] > 0):
                done = True
                #break

        if not done:
            if action == 0: # FORWARD
                reward = 5
            else:
                reward = 1
        else:
            reward = -200 

        state = discretized_ranges

        return state, reward, done, {}

    def _reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except rospy.ServiceException, e:
            print ("/gazebo/reset_world service call failed")

        self._takeoff(1)

        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        #simplify ranges - discretize
        discretized_ranges = []
        discretized_ranges_amount = 5
        min_range = 0.2 #collision

        done = False

        mod = (len(data.ranges) / discretized_ranges_amount)
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf'):
                    discretized_ranges.append(int(data.range_max))
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
                    #discretized_ranges.append(round(data.ranges[i] * 2) / 2)

        state = discretized_ranges

        return state
