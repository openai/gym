import gym
import rospy
#import roslaunch
import os
import subprocess

from os import path
from std_srvs.srv import Empty

class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, launchfile):

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym', anonymous=True)

        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets","launch", launchfile)
        if not path.exists(fullpath):
            raise IOError("File "+fullpath+" does not exist")

        #uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        #launch = roslaunch.parent.ROSLaunchParent(uuid, fullpath)
        #launch.start()

        subprocess.Popen(["roslaunch",fullpath])

        print "Gazebo launched!"

    def _step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def _reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            reset_proxy.call()
        except rospy.ServiceException, e:
            print "/gazebo/reset_simulation service call failed"


    def _render(self, mode="human", close=False):

        # Open/Close GUI

        if close:
            #Close gzclient
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                subprocess.call("kill `pidof gzclient`")
            else:
                print "gzclient is not running"
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.call("gzclient")
        else:
            print "gzclient already running"

    def _close(self):

        # Kill gzclient, gzserver and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')

        if gzclient_count > 0:
            subprocess.call("kill `pidof gzclient`")
        if gzserver_count > 0:
            subprocess.call("kill `pidof gzserver`")
        if roscore_count > 0:
            subprocess.call("kill `pidof roscore`")


    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass
    def _seed(self):
        
        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)  
        pass
