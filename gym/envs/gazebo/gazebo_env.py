import gym
import rospy
#import roslaunch
import os
import subprocess

from os import path


class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    
    def __init__(self, launchfile):

        # Launch the simulation with the given launchfile name
        self.node = rospy.init_node('gym', anonymous=True)

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

    def _spawn_robot(self):

        # TODO
        # Spawn robot
        # Optional
        # Another option is to spawn the robot in the __init__ method
        pass

    def _step(self, action):

        # TODO
        # Perform a step in gazebo. E.g. move the robot
        pass
    def _reset(self):

        # TODO
        # Reset world/simulation
        pass
    def _render(self, episodes):

        # Open GUI (if it's not allready opened?)
        # episodes = number of episodes that GUI is going to be opened. Another option is to use _close to close the gui
        pass
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