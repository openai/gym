import gym

class GazeboEnv(gym.Env):

	def __init__(self, world):

		# TODO
		# Launch the simulation with the given world/launchfile

	def _spawn_robot(self):

		# TODO
		# Spawn robot
		# Optional
		# Another option is to spawn the robot in the __init__ method

	def _step(self, action):

		# TODO
		# Perform a step in gazebo. E.g. move the robot

	def _reset(self):

		# TODO
		# Reset world/simulation

	def _render(self, episodes):

		# Open GUI (if it's not allready opened?)
		# episodes = number of episodes that GUI is going to be opened. Another option is to use _close to close the gui

	def _close(self):

		# TODO
		# From OpenAI API: Perform any necessary cleanup

	def _configure(self):

		# TODO
		# From OpenAI API: Provides runtime configuration to the enviroment
		# Maybe set the Real Time Factor?

	def _seed(self):
		
		# TODO
		# From OpenAI API: Sets the seed for this env's random number generator(s)	