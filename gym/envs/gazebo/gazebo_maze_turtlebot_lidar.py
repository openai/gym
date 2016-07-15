import gym
import rospy
import roslaunch
from gym.envs.gazebo import gazebo_env

class GazeboMazeTurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):

        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboMazeTurtlebotLidar_v0.launch")

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