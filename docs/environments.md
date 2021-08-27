# Environments

This is a list of Gym environments, including those packaged with Gym, official OpenAI environments, and third party environment.

For information on creating your own environment, see [Creating your own Environment](creating-environments.md).

## Included Environments

The code for each environment group is housed in its own subdirectory
[gym/envs](https://github.com/openai/gym/blob/master/gym/envs). The
specification of each task is in
[gym/envs/\_\_init\_\_.py](https://github.com/openai/gym/blob/master/gym/envs/__init__.py).
It's worth browsing through both.

### Atari

The Atari environments are a variety of Atari video games. If you didn't
do the full install, you can install dependencies via `pip install -e
'.[atari]'` (you'll need `cmake` installed) and then get started as
follows:

``` python
import gym
env = gym.make('SpaceInvaders-v4')
env.reset()
env.render()
```

This will install `atari-py`, which automatically compiles the [Arcade
Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment#:~:text=The%20Arcade%20Learning%20Environment%20(ALE)%20is%20a%20simple%20object%2D,of%20emulation%20from%20agent%20design.). This
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
env = gym.make('CartPole-v1')
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

These environments also use [MuJoCo](http://www.mujoco.org/). You'll have to also run `pip install -e '.[robotics]'` if
you didn't do the full install.

``` python
import gym
env = gym.make('HandManipulateBlock-v2')
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
env = gym.make('FrozenLake-v1')
env.reset()
env.render()
```

## OpenAI Environments

### Procgen

16 simple-to-use procedurally-generated gym environments which provide a direct measure of how quickly a reinforcement learning agent learns generalizable skills. The environments run at high speed (thousands of steps per second) on a single core.

Learn more here: https://github.com/openai/procgen

### Gym-Retro

Gym Retro lets you turn classic video games into Gym environments for reinforcement learning and comes with integrations for ~1000 games. It uses various emulators that support the Libretro API, making it fairly easy to add new emulators.

Learn more here: https://github.com/openai/retro

### Roboschool (DEPRECATED)

**We recommend using the [PyBullet Robotics Environments](#pybullet-robotics-environments) instead**

3D physics environments like Mujoco environments but uses the Bullet physics engine and does not require a commercial license.

Learn more here: https://github.com/openai/roboschool

## Third Party Environments

The gym comes prepackaged with many many environments. It's this common API around many environments that makes Gym so great. Here we will list additional environments that do not come prepacked with the gym. Submit another to this list via a pull-request.

### gym-algorithmic

These are a variety of algorithmic tasks, such as learning to copy a sequence, present in Gym prior to Gym 0.20.0.

Learn more here: https://github.com/Rohan138/gym-algorithmic

### gym-spoof

Spoof, otherwise known as "The 3-coin game", is a multi-agent (2 player), imperfect-information, zero-sum game. 

Learn more here: https://github.com/MouseAndKeyboard/gym-spoof

Platforms: Windows, Mac, Linux

### PyBullet Robotics Environments

3D physics environments like the Mujoco environments but uses the Bullet physics engine and does not require a commercial license.  Works on Mac/Linux/Windows.

Learn more here: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.wz5to0x8kqmr

### Obstacle Tower

3D procedurally generated tower where you have to climb to the highest level possible

Learn more here: https://github.com/Unity-Technologies/obstacle-tower-env

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

### gym-anytrading: Environments for trading markets

AnyTrading is a collection of OpenAI Gym environments for reinforcement learning-based trading algorithms with a great focus on simplicity, flexibility, and comprehensiveness.

Learn more here: https://github.com/AminHP/gym-anytrading

### GymGo: The Board Game Go

An implementation of the board game Go

Learn more here: https://github.com/aigagror/GymGo 

### gym-electric-motor: Intelligent control of electric drives

An environment for simulating a wide variety of electric drives taking into account different types of electric motors and converters. Control schemes can be continuous, yielding a voltage duty cycle, or discrete, determining converter switching states directly.

Learn more here: https://github.com/upb-lea/gym-electric-motor

### NASGym: gym environment for Neural Architecture Search (NAS)

The environment is fully-compatible with the OpenAI baselines and exposes a NAS environment following the Neural Structure Code of [BlockQNN: Efficient Block-wise Neural Network Architecture Generation](https://arxiv.org/abs/1808.05584). Under this setting, a Neural Network (i.e. the state for the reinforcement learning agent) is modeled as a list of NSCs, an action is the addition of a layer to the network, and the reward is the accuracy after the early-stop training. The datasets considered so far are the CIFAR-10 dataset (available by default) and the meta-dataset (has to be manually downloaded as specified in [this repository](https://github.com/gomerudo/meta-dataset)).

Learn more here: https://github.com/gomerudo/nas-env

### gym-jiminy: training Robots in Jiminy

gym-jiminy presents an extension of the initial OpenAI gym for robotics using Jiminy, an extremely fast and light weight simulator for poly-articulated systems using Pinocchio for physics evaluation and Meshcat for web-based 3D rendering.

Learn more here: https://github.com/Wandercraft/jiminy

### highway-env: Tactical Decision-Making for Autonomous Driving

An environment for behavioural planning in autonomous driving, with an emphasis on high-level perception and decision rather than low-level sensing and control. The difficulty of the task lies in understanding the social interactions with other drivers, whose behaviours are uncertain. Several scenes are proposed, such as highway, merge, intersection and roundabout.

Learn more here: https://github.com/eleurent/highway-env

### gym-carla: Gym Wrapper for CARLA Driving Simulator

gym-carla provides a gym wrapper for the [CARLA simulator](http://carla.org/), which is a realistic 3D simulator for autonomous driving research. The environment includes a virtual city with several surrounding vehicles running around. Multiple source of observations are provided for the ego vehicle, such as front-view camera image, lidar point cloud image, and birdeye view semantic mask. Several applications have been developed based on this wrapper, such as deep reinforcement learning for end-to-end autonomous driving.

Learn more here: https://github.com/cjy1992/gym-carla

### openmodelica-microgrid-gym: Intelligent control of microgrids 

The OpenModelica Microgrid Gym (OMG) package is a software toolbox for the simulation and control optimization of microgrids based on energy conversion by power electronic converters.

Learn more here: https://github.com/upb-lea/openmodelica-microgrid-gym

### RubiksCubeGym: OpenAI Gym environments for various twisty puzzles

The RubiksCubeGym package provides environments for twisty puzzles with  multiple reward functions to help simluate the methods used by humans.

Learn more here: https://github.com/DoubleGremlin181/RubiksCubeGym

### SlimeVolleyGym: A simple environment for single and multi-agent reinforcement learning

A simple environment for benchmarking single and multi-agent reinforcement learning algorithms on a clone of Slime Volleyball game. Only dependencies are gym and numpy. Both state and pixel observation environments are available. The motivation of this environment is to easily enable trained agents to play against each other, and also facilitate the training of agents directly in a multi-agent setting, thus adding an extra dimension for evaluating an agent's performance.

Learn more here: https://github.com/hardmaru/slimevolleygym

### Gridworld: A simple 2D grid environment

The Gridworld package provides grid-based environments to help simulate the results for model-based reinforcement learning algorithms. Initial release supports single agent system only. Some features in this version of software have become obsolete. New features are being added in the software like windygrid environment.

Learn more here: https://github.com/addy1997/Gridworld

### gym-goddard: Goddard's Rocket Problem

An environment for simulating the classical optimal control problem where the thrust of a vertically ascending rocket shall be determined such that it reaches the maximum possible altitude, while being subject to varying aerodynamic drag, gravity and mass. 

Learn more here: https://github.com/osannolik/gym-goddard

### gym-pybullet-drones: Learning Quadcopter Control

A simple environment using [PyBullet](https://github.com/bulletphysics/bullet3) to simulate the dynamics of a [Bitcraze Crazyflie 2.x](https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf) nanoquadrotor

Learn more here: https://github.com/JacopoPan/gym-pybullet-drones

### gym-derk: GPU accelerated MOBA environment

This is a 3v3 MOBA environment where you train creatures to figth each other. It runs entirely on the GPU so you can easily have hundreds of instances running in parallel. There are around 15 items for the creatures, 60 "senses", 5 actions, and ~23 tweakable rewards. It's also possible to benchmark an agent against other agents online. It's available for free for training for personal use, and otherwise costs money; see licensing details on the website.

More here: https://gym.derkgame.com

### gym-abalone: A two-player abstract strategy board game

An implementation of the board game Abalone.

Learn more here: https://github.com/towzeur/gym-abalone

### gym-adserver: Environment for online advertising

An environment that implements a typical [multi-armed bandit scenario](https://en.wikipedia.org/wiki/Multi-armed_bandit) where an [ad server](https://en.wikipedia.org/wiki/Ad_serving) must select the best advertisement to be displayed in a web page. Some example agents are included: Random, epsilon-Greedy, Softmax, and UCB1.

Learn more here: https://github.com/falox/gym-adserver

### gym-autokey: Automated rule-based deductive program verification

An environment for automated rule-based deductive program verification in the KeY verification system.

Learn more here: https://github.com/Flunzmas/gym-autokey

### gym-riverswim: A hard-exploration environment

A simple environment for benchmarking reinforcement learning exploration techniques in a simplified setting.

Learn more here: https://github.com/erfanMhi/gym-riverswim

### gym-ccc: Continuous classic control environments

Environments that extend gym's classic control and add more problems.
These environments have features useful for non-RL controllers.

The main highlights are:
1) non normalized observation corresponding directly to the dynamical state
2) normalized observation with dynamical state captured in `info['state']`
3) action spaces are continuous
4) system parameters (mass, length, etc.) can be specified
5) reset function (to specify initial conditions) can be specified.

Learn more here: https://github.com/acxz/gym-ccc

### NLPGym: A toolkit to develop RL agents to solve NLP tasks

[NLPGym](https://arxiv.org/pdf/2011.08272v1.pdf) provides interactive environments for standard NLP tasks such as sequence tagging, question answering, and sequence classification. Users can easily customize the tasks with their own datasets, observations, featurizers and reward functions.

Learn more here: https://github.com/rajcscw/nlp-gym

### math-prog-synth-env

In our paper "A Reinforcement Learning Environment for Mathematical Reasoning via Program Synthesis" we convert the DeepMind Mathematics Dataset into an RL environment based around program synthesis.

Learn more here: https://github.com/JohnnyYeeee/math_prog_synth_env , https://arxiv.org/abs/2107.07373

### VirtualTaobao: Environment of online recommendation

An environment for online recommendation, where customers are learned from Taobao.com, one of the world's largest e-commerce platform.

Learn more here: https://github.com/eyounx/VirtualTaobao/

### gym-recsys: Customizable RecSys Simulator for OpenAI Gym

This package describes an OpenAI Gym interface for creating a simulation environment of reinforcement learning-based recommender systems (RL-RecSys). The design strives for simple and flexible APIs to support novel research.

Learn more here: https://github.com/zuoxingdong/gym-recsys

### QASGym: gym environment for Quantum Architecture Search (QAS)

This a list of environments for quantum architecture search following the description in [Quantum Architecture Search via Deep Reinforcement Learning](https://arxiv.org/abs/2104.07715). The agent design the quantum circuit by taking actions in the environment. Each action corresponds to a gate applied on some wires. The goal is to build a circuit U such that generates the target n-qubit quantum state that belongs to the environment and hidden from the agent. The circuits are built using [Google QuantumAI Cirq](https://quantumai.google/cirq). 

Learn more here: https://github.com/qdevpsi3/quantum-arch-search

### robo-gym: Environments for Real and Simulated Robots

robo-gym provides a collection of reinforcement learning environments involving robotic tasks applicable in both simulation and real world robotics. 

Learn more here: https://github.com/jr-robotics/robo-gym

### gym-xiangqi: Xiangqi - The Chinese Chess Game

A reinforcement learning environment of Xiangqi, the Chinese Chess game.

Learn more here: https://github.com/tanliyon/gym-xiangqi

### anomalous_rl_envs: Gym environments with anomaly injection

A set of environments from control tasks: Acrobot, CartPole, and LunarLander with various types of anomalies injected into them. It could be very useful to study the behavior and robustness of a policy.

Learn more here: https://github.com/modanesh/anomalous_rl_envs

### stable-retro

Fork of gym-retro with additional games, states, scenarios, etc. Open to PRs of additional games, features and plateforms since gym-retro is in maintenance mode.

https://github.com/MatPoliquin/stable-retro

### CompilerGym

Reinforcement learning environments for compiler optimization tasks, such as LLVM phase ordering, GCC flag tuning, and CUDA loop nest code generation.

Learn more here: https://github.com/facebookresearch/CompilerGym

### LongiControl

An environment for the stochastic longitudinal control of an electric vehicle.
It is intended to be a descriptive and comprehensible example for a continuous real-world problem within the field of autonomous driving.

Learn more here: https://github.com/dynamik1703/gym_longicontrol

### safe-control-gym

PyBullet-based CartPole and Quadrotor environments—with [CasADi](https://web.casadi.org) (symbolic) *a priori* dynamics and constraints—for learning-based control and model-based reinforcement learning.

Learn more here: https://github.com/utiasDSL/safe-control-gym
