# Environments

This is a list of Gym environments, including those packaged with Gym, official OpenAI environments, and third party environment.

For information on creating your own environment, see [Creating your own Environment](creating-environments.md).

## Included Environments

The code for each environment group is housed in its own subdirectory
[gym/envs](https://github.com/openai/gym/blob/master/gym/envs). The
specification of each task is in
[gym/envs/\_\_init\_\_.py](https://github.com/openai/gym/blob/master/gym/envs/__init__.py).
It's worth browsing through both.

### Algorithmic

These are a variety of algorithmic tasks, such as learning to copy a
sequence.

``` python
import gym
env = gym.make('Copy-v0')
env.reset()
env.render()
```

### Atari

The Atari environments are a variety of Atari video games. If you didn't
do the full install, you can install dependencies via `pip install -e
'.[atari]'` (you'll need `cmake` installed) and then get started as
follows:

``` python
import gym
env = gym.make('SpaceInvaders-v0')
env.reset()
env.render()
```

This will install `atari-py`, which automatically compiles the [Arcade
Learning Environment](http://www.arcadelearningenvironment.org/). This
can take quite a while (a few minutes on a decent laptop), so just be
prepared.

### Box2d

Box2d is a 2D physics engine. You can install it via `pip install -e
'.[box2d]'` and then get started as follows:

``` python
import gym
env = gym.make('LunarLander-v2')
env.reset()
env.render()
```

### Classic control

These are a variety of classic control tasks, which would appear in a
typical reinforcement learning textbook. If you didn't do the full
install, you will need to run `pip install -e '.[classic_control]'` to
enable rendering. You can get started with them via:

``` python
import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()
```

### MuJoCo

[MuJoCo](http://www.mujoco.org/) is a physics engine which can do very
detailed efficient simulations with contacts. It's not open-source, so
you'll have to follow the instructions in
[mujoco-py](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key)
to set it up. You'll have to also run `pip install -e '.[mujoco]'` if
you didn't do the full install.

``` python
import gym
env = gym.make('Humanoid-v2')
env.reset()
env.render()
```

### Robotics

[MuJoCo](http://www.mujoco.org/) is a physics engine which can do very
detailed efficient simulations with contacts and we use it for all
robotics environments. It's not open-source, so you'll have to follow
the instructions in
[mujoco-py](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key)
to set it up. You'll have to also run `pip install -e '.[robotics]'` if
you didn't do the full install.

``` python
import gym
env = gym.make('HandManipulateBlock-v0')
env.reset()
env.render()
```

You can also find additional details in the accompanying [technical
report](https://arxiv.org/abs/1802.09464) and [blog
post](https://blog.openai.com/ingredients-for-robotics-research/). If
you use these environments, you can cite them as follows:

    @misc{1802.09464,
      Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
      Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
      Year = {2018},
      Eprint = {arXiv:1802.09464},
    }

### Toy text

Toy environments which are text-based. There's no extra dependency to
install, so to get started, you can just do:

``` python
import gym
env = gym.make('FrozenLake-v0')
env.reset()
env.render()
```

## OpenAI Environments

### Roboschool

3D physics environments like Mujoco environments but uses the Bullet physics engine and does not require a commercial license.

Learn more here: https://github.com/openai/roboschool

### Gym-Retro

Gym Retro lets you turn classic video games into Gym environments for reinforcement learning and comes with integrations for ~1000 games. It uses various emulators that support the Libretro API, making it fairly easy to add new emulators.

Learn more here: https://github.com/openai/retro

## Third Party Environments

The gym comes prepackaged with many many environments. It's this common API around many environments that makes Gym so great. Here we will list additional environments that do not come prepacked with the gym. Submit another to this list via a pull-request. 

### Pybullet Robotics Environments

3D physics environments like the Mujoco environments but uses the Bullet physics engine and does not require a commercial license.  Works on Mac/Linux/Windows.

Learn more here: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.wz5to0x8kqmr

### Obstacle Tower

3D procedurally generated tower where you have to climb to the highest level possible

Learn more here: https://github.com/Unity-Technologies/obstacle-tower-challenge

Platforms: Windows, Mac, Linux

### PGE: Parallel Game Engine

PGE is a FOSS 3D engine for AI simulations, and can interoperate with the Gym. Contains environments with modern 3D graphics, and uses Bullet for physics.

Learn more here: https://github.com/222464/PGE

### gym-inventory: Inventory Control Environments

gym-inventory is a single agent domain featuring discrete state and action spaces that an AI agent might encounter in inventory control problems. 

Learn more here: https://github.com/paulhendricks/gym-inventory

### gym-gazebo: training Robots in Gazebo

gym-gazebo presents an extension of the initial OpenAI gym for robotics using ROS and Gazebo, an advanced 3D modeling and
rendering  tool.

Learn more here: https://github.com/erlerobot/gym-gazebo/

### gym-maze: 2D maze environment
A simple 2D maze environment where an agent finds its way from the start position to the goal. 

Learn more here: https://github.com/tuzzer/gym-maze/

### osim-rl: Musculoskeletal Models in OpenSim

A human musculoskeletal model and a physics-based simulation environment where you can synthesize physically and physiologically accurate motion. One of the environments built in this framework is a competition environment for a NIPS 2017 challenge.

Learn more here: https://github.com/stanfordnmbl/osim-rl

### gym-minigrid: Minimalistic Gridworld Environment

A minimalistic gridworld environment. Seeks to minimize software dependencies, be easy to extend and deliver good performance for faster training.

Learn more here: https://github.com/maximecb/gym-minigrid

### gym-miniworld: Minimalistic 3D Interior Environment Simulator 

MiniWorld is a minimalistic 3D interior environment simulator for reinforcement learning & robotics research. It can be used to simulate environments with rooms, doors, hallways and various objects (eg: office and home environments, mazes). MiniWorld can be seen as an alternative to VizDoom or DMLab. It is written 100% in Python and designed to be easily modified or extended.

Learn more here: https://github.com/maximecb/gym-miniworld

### gym-sokoban: 2D Transportation Puzzles

The environment consists of transportation puzzles in which the player's goal is to push all boxes on the warehouse's storage locations.
The advantage of the environment is that it generates a new random level every time it is initialized or reset, which prevents over fitting to predefined levels.

Learn more here: https://github.com/mpSchrader/gym-sokoban

### gym-duckietown: Lane-Following Simulator for Duckietown

A lane-following simulator built for the [Duckietown](http://duckietown.org/) project (small-scale self-driving car course).

Learn more here: https://github.com/duckietown/gym-duckietown

### GymFC: A flight control tuning and training framework 

GymFC is a modular framework for synthesizing neuro-flight controllers. The
architecture integrates digital twinning concepts to provide seamless transfer
of trained policies to hardware. The OpenAI environment has been used to
generate policies for the worlds first open source neural network flight
control firmware [Neuroflight](https://github.com/wil3/neuroflight).

Learn more here: https://github.com/wil3/gymfc/
