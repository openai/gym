import gym
import os
import rospy
import roslaunch
import subprocess
import time
import numpy as np

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding

from mavros_msgs.msg import OverrideRCIn
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64

from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from std_srvs.srv import Empty


class GazeboMazeErleRoverLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):

        self._launch_apm()
        RED = '\033[91m'
        BOLD = '\033[1m'
        ENDC = '\033[0m'
        LINE = "%s%s##############################################################################%s" % (RED, BOLD, ENDC)
        msg = "\n%s\n" % (LINE)
        msg += "%sLoad Erle-Rover parameters in MavProxy console (sim_vehicle.sh):%s\n\n" % (BOLD, ENDC)
        msg += "MAV> param load %s\n\n" % (str(os.environ["ERLE_ROVER_PARAM_PATH"]))
        msg += "%sThen, press <Enter> here to launch Gazebo...%s\n\n%s" % (BOLD, ENDC,  LINE)
        self._pause(msg)

        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboMazeErleRoverLidar_v0.launch")

        self.pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=10)

        self.action_space = spaces.Discrete(3) #F,L,R
        #self.observation_space = spaces.Box(low=0, high=20) #laser values
        self.reward_range = (-np.inf, np.inf)

        self.gazebo_step_size = long(200)


        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.mode_proxy = rospy.ServiceProxy('mavros/set_mode', SetMode)

        time.sleep(10) # Wait for gzserver to launch

        # Set MANUAL mode
        rospy.wait_for_service('mavros/set_mode')
        try:
            self.mode_proxy(0,'GUIDED')
        except (rospy.ServiceException) as e:
            print ("mavros/set_mode service call failed: %s"%e)

        print ("Waiting for mavros...")
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/mavros/global_position/rel_alt', Float64, timeout=5)
            except:
                pass

        self._seed()

    def _launch_apm(self):
        sim_vehicle_sh = str(os.environ["ARDUPILOT_PATH"]) + "/Tools/autotest/sim_vehicle.sh"
        subprocess.Popen(["xterm","-e",sim_vehicle_sh,"-j4","-f","Gazebo","-v","APMrover2"])

    def _pause(self, msg):
        programPause = raw_input(str(msg))


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _laser_state(self, action):


        return discretized_ranges, done

    def step(self, action):

        msg = OverrideRCIn()

        if action == 0: #FORWARD
            msg.channels[0] = 1500 # Yaw
            msg.channels[2] = 1900 # Throttle
        elif action == 1: #LEFT
            msg.channels[0] = 1100 # Yaw
            msg.channels[2] = 1900 # Throttle
        elif action == 2: #RIGHT
            msg.channels[0] = 1900 # Yaw
            msg.channels[2] = 1900 # Throttle

        msg.channels[1] = 0
        msg.channels[3] = 0
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
        min_range = 1.5 #collision

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

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_world service call failed")

        # Set MANUAL mode
        rospy.wait_for_service('mavros/set_mode')
        try:
            self.mode_proxy(0,'MANUAL')
        except (rospy.ServiceException) as e:
            print ("mavros/set_mode service call failed: %s"%e)

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
