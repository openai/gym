from setuptools import setup
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym-gazebo'))

setup(name='gym-gazebo',
      version='0.0.1',
      install_requires=['gym>=0.2.3'],
      description='The OpenAI Gym for robotics: A toolkit for developing and comparing your reinforcement learning agents using Gazebo and ROS.',
      url='https://github.com/erlerobot/gym',
      author='Erle Robotics',
      package_data={'gym-gazebo': ['envs/assets/launch/*.launch', 'envs/assets/worlds/*']},
)
