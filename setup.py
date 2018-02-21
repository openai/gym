from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym'))
from version import VERSION

# Environment-specific dependencies.
extras = {
  'atari': ['atari_py>=0.1.1', 'Pillow', 'PyOpenGL'],
  'box2d': ['Box2D-kengz'],
  'classic_control': ['PyOpenGL'],
  'mujoco': ['mujoco_py>=1.50', 'imageio'],
}

# Meta dependency groups.
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

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
          'numpy>=1.10.4', 'requests>=2.0', 'six', 'pyglet>=1.2.0',
      ],
      extras_require=extras,
      package_data={'gym': ['envs/mujoco/assets/*.xml', 'envs/classic_control/assets/*.png']},
      tests_require=['pytest', 'mock'],
)
