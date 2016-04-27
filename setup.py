from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym'))
from version import VERSION

setup(name='gym',
      version=VERSION,
      description='The OpenAI Gym: A toolkit for developing and comparing your reinforcement learning agents.',
      url='https://github.com/openai/gym',
      author='OpenAI',
      author_email='gym@openai.com',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('gym')],
      zip_safe=False,
      install_requires=[
          'numpy>=1.10.4', 'requests', 'six'
      ],
      extras_require={
          'all': ['atari_py>=0.0.14', 'Pillow', 'pyglet',
                  'pachi-py>=0.0.16',
                  'mujoco_py>=0.4.0', 'imageio'],

          # Environment-specific dependencies. Keep these in sync with
          # 'all'!
          'atari': ['atari_py>=0.0.14', 'Pillow', 'pyglet'],
          'board_game' : ['pachi-py>=0.0.16'],
          'classic_control': ['pyglet'],
          'mujoco': ['mujoco_py>=0.4.0', 'imageio'],
      },
      tests_require=['nose2', 'mock'],
)
