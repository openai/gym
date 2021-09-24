# Third Party Environments

## Video Game Environments

### gym-derk: GPU accelerated MOBA environment

https://gym.derkgame.com

This is a 3v3 MOBA environment where you train creatures to fight each other. It runs entirely on the GPU so you can easily have hundreds of instances running in parallel. There are around 15 items for the creatures, 60 "senses", 5 actions, and roughly 23 tweakable rewards. It's also possible to benchmark an agent against other agents online. It's available for free for training for personal use, and otherwise costs money; see licensing details on the website

### MineRL

https://github.com/minerllabs/minerl

Gym interface with Minecraft game, focused on a specific sparse reward challenge

### Procgen

https://github.com/openai/procgen

16 simple-to-use procedurally-generated gym environments which provide a direct measure of how quickly a reinforcement learning agent learns generalizable skills. The environments run at high speed (thousands of steps per second) on a single core.

### SlimeVolleyGym: A simple environment for single and multi-agent reinforcement learning

https://github.com/hardmaru/slimevolleygym

A simple environment for benchmarking single and multi-agent reinforcement learning algorithms on a clone of Slime Volleyball game. Only dependencies are gym and numpy. Both state and pixel observation environments are available. The motivation of this environment is to easily enable trained agents to play against each other, and also facilitate the training of agents directly in a multi-agent setting, thus adding an extra dimension for evaluating an agent's performance.

### stable-retro

https://github.com/MatPoliquin/stable-retro

Supported fork of gym-retro with additional games, states, scenarios, etc. Open to PRs of additional games, features and plateforms since gym-retro is no longer maintained

### Unity ML Agents

https://github.com/Unity-Technologies/ml-agents

Gym wrappers for arbitrary and premade environments with the Unity game engine.

## Classic Environments (board, card, etc. games)

### gym-abalone: A two-player abstract strategy board game

https://github.com/towzeur/gym-abalone

An implementation of the board game Abalone.

### gym-spoof

https://github.com/MouseAndKeyboard/gym-spoof

Spoof, otherwise known as "The 3-coin game", is a multi-agent (2 player), imperfect-information, zero-sum game.

### gym-xiangqi: Xiangqi - The Chinese Chess Game

https://github.com/tanliyon/gym-xiangqi

A reinforcement learning environment of Xiangqi, the Chinese Chess game.

### RubiksCubeGym

https://github.com/DoubleGremlin181/RubiksCubeGym

The RubiksCubeGym package provides environments for twisty puzzles with  multiple reward functions to help simluate the methods used by humans.

## Robotics Environments

### GymFC: A flight control tuning and training framework

https://github.com/wil3/gymfc/

GymFC is a modular framework for synthesizing neuro-flight controllers. The architecture integrates digital twinning concepts to provide seamless transfer of trained policies to hardware. The OpenAI environment has been used to generate policies for the worlds first open source neural network flight control firmware [Neuroflight](https://github.com/wil3/neuroflight).

### gym-gazebo

https://github.com/erlerobot/gym-gazebo/

gym-gazebo presents an extension of the initial OpenAI gym for robotics using ROS and Gazebo, an advanced 3D modeling and
rendering  tool.

### gym-goddard: Goddard's Rocket Problem

https://github.com/osannolik/gym-goddard

An environment for simulating the classical optimal control problem where the thrust of a vertically ascending rocket shall be determined such that it reaches the maximum possible altitude, while being subject to varying aerodynamic drag, gravity and mass.

### gym-jiminy: training Robots in Jiminy

https://github.com/Wandercraft/jiminy

gym-jiminy presents an extension of the initial OpenAI gym for robotics using Jiminy, an extremely fast and light weight simulator for poly-articulated systems using Pinocchio for physics evaluation and Meshcat for web-based 3D rendering.

### gym-miniworld

https://github.com/maximecb/gym-miniworld

MiniWorld is a minimalistic 3D interior environment simulator for reinforcement learning & robotics research. It can be used to simulate environments with rooms, doors, hallways and various objects (eg: office and home environments, mazes). MiniWorld can be seen as an alternative to VizDoom or DMLab. It is written 100% in Python and designed to be easily modified or extended.

### gym-pybullet-drones

https://github.com/JacopoPan/gym-pybullet-drones

A simple environment using [PyBullet](https://github.com/bulletphysics/bullet3) to simulate the dynamics of a [Bitcraze Crazyflie 2.x](https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf) nanoquadrotor.

### MarsExplorer

https://github.com/dimikout3/MarsExplorer

Mars Explorer is an openai-gym compatible environment designed and developed as an initial endeavor to bridge the gap between powerful Deep Reinforcement Learning methodologies and the problem of exploration/coverage of an unknown terrain.

### panda-gym 

https://github.com/qgallouedec/panda-gym/

PyBullet based simulations of a robotic arm moving objects.

### PyBullet Robotics Environments

https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.wz5to0x8kqmr

3D physics environments like the Mujoco environments but uses the Bullet physics engine and does not require a commercial license.  Works on Mac/Linux/Windows.

### robo-gym

https://github.com/jr-robotics/robo-gym

robo-gym provides a collection of reinforcement learning environments involving robotic tasks applicable in both simulation and real world robotics.

### Offworld-gym

https://github.com/offworld-projects/offworld-gym

Gym environments that let you control physics robotics in a laboratory via the internet.

## Autonomous Driving and Traffic Control Environments

### gym-carla

https://github.com/cjy1992/gym-carla

gym-carla provides a gym wrapper for the [CARLA simulator](http://carla.org/), which is a realistic 3D simulator for autonomous driving research. The environment includes a virtual city with several surrounding vehicles running around. Multiple source of observations are provided for the ego vehicle, such as front-view camera image, lidar point cloud image, and birdeye view semantic mask. Several applications have been developed based on this wrapper, such as deep reinforcement learning for end-to-end autonomous driving.

### gym-duckietown

https://github.com/duckietown/gym-duckietown

A lane-following simulator built for the [Duckietown](http://duckietown.org/) project (small-scale self-driving car course).

### gym-electric-motor

https://github.com/upb-lea/gym-electric-motor

An environment for simulating a wide variety of electric drives taking into account different types of electric motors and converters. Control schemes can be continuous, yielding a voltage duty cycle, or discrete, determining converter switching states directly.

### highway-env

https://github.com/eleurent/highway-env

An environment for behavioural planning in autonomous driving, with an emphasis on high-level perception and decision rather than low-level sensing and control. The difficulty of the task lies in understanding the social interactions with other drivers, whose behaviours are uncertain. Several scenes are proposed, such as highway, merge, intersection and roundabout.

### LongiControl

https://github.com/dynamik1703/gym_longicontrol

An environment for the stochastic longitudinal control of an electric vehicle. It is intended to be a descriptive and comprehensible example for a continuous real-world problem within the field of autonomous driving.

### sumo-rl

https://github.com/LucasAlegre/sumo-rl

Gym wrapper for various environments in the Sumo traffic simulator

## Other Environments

### anomalous_rl_envs

https://github.com/modanesh/anomalous_rl_envs

A set of environments from control tasks: Acrobot, CartPole, and LunarLander with various types of anomalies injected into them. It could be very useful to study the behavior and robustness of a policy.

### CompilerGym

https://github.com/facebookresearch/CompilerGym

Reinforcement learning environments for compiler optimization tasks, such as LLVM phase ordering, GCC flag tuning, and CUDA loop nest code generation.

### Gridworld

https://github.com/addy1997/Gridworld

The Gridworld package provides grid-based environments to help simulate the results for model-based reinforcement learning algorithms. Initial release supports single agent system only. Some features in this version of software have become obsolete. New features are being added in the software like windygrid environment.

### gym-adserve

https://github.com/falox/gym-adserver

An environment that implements a typical [multi-armed bandit scenario](https://en.wikipedia.org/wiki/Multi-armed_bandit) where an [ad server](https://en.wikipedia.org/wiki/Ad_serving) must select the best advertisement to be displayed in a web page. Some example agents are included: Random, epsilon-Greedy, Softmax, and UCB1.

### gym-algorithmic

https://github.com/Rohan138/gym-algorithmic

These are a variety of algorithmic tasks, such as learning to copy a sequence, present in Gym prior to Gym 0.20.0.

### gym-anytrading

https://github.com/AminHP/gym-anytrading

AnyTrading is a collection of OpenAI Gym environments for reinforcement learning-based trading algorithms with a great focus on simplicity, flexibility, and comprehensiveness.

### gym-autokey

https://github.com/Flunzmas/gym-autokey

An environment for automated rule-based deductive program verification in the KeY verification system.

### gym-ccc

https://github.com/acxz/gym-ccc

Environments that extend gym's classic control and add many new features including continuous action spaces.

### gym-games

https://github.com/qlan3/gym-games

Gym implementations of the MinAtar games, various PyGame Learning Environment games, and various custom exploration games

### gym-inventory

https://github.com/paulhendricks/gym-inventory

gym-inventory is a single agent domain featuring discrete state and action spaces that an AI agent might encounter in inventory control problems.

### gym-maze

https://github.com/tuzzer/gym-maze/

A simple 2D maze environment where an agent finds its way from the start position to the goal.

### gym-mtsim

https://github.com/AminHP/gym-mtsim

MtSim is a general-purpose, flexible, and easy-to-use simulator alongside an OpenAI Gym trading environment for MetaTrader 5 trading platform.

### gym-legacy-toytext

https://github.com/Rohan138/gym-legacy-toytext

These are the unused toy-text environments present in Gym prior to Gym 0.20.0.

### gym-riverswim

https://github.com/erfanMhi/gym-riverswim

A simple environment for benchmarking reinforcement learning exploration techniques in a simplified setting. Hard exploration.

### gym-recsys

https://github.com/zuoxingdong/gym-recsys

This package describes an OpenAI Gym interface for creating a simulation environment of reinforcement learning-based recommender systems (RL-RecSys). The design strives for simple and flexible APIs to support novel research.

### gym-sokoban

https://github.com/mpSchrader/gym-sokoban

2D Transportation Puzzles. The environment consists of transportation puzzles in which the player's goal is to push all boxes on the warehouse's storage locations. The advantage of the environment is that it generates a new random level every time it is initialized or reset, which prevents over fitting to predefined levels.

### math-prog-synth-env

https://github.com/JohnnyYeeee/math_prog_synth_env

In our paper "A Reinforcement Learning Environment for Mathematical Reasoning via Program Synthesis" we convert the DeepMind Mathematics Dataset into an RL environment based around program synthesis.https://arxiv.org/abs/2107.07373

### NASGym

https://github.com/gomerudo/nas-env

The environment is fully-compatible with the OpenAI baselines and exposes a NAS environment following the Neural Structure Code of [BlockQNN: Efficient Block-wise Neural Network Architecture Generation](https://arxiv.org/abs/1808.05584). Under this setting, a Neural Network (i.e. the state for the reinforcement learning agent) is modeled as a list of NSCs, an action is the addition of a layer to the network, and the reward is the accuracy after the early-stop training. The datasets considered so far are the CIFAR-10 dataset (available by default) and the meta-dataset (has to be manually downloaded as specified in [this repository](https://github.com/gomerudo/meta-dataset)).

### NLPGym: A toolkit to develop RL agents to solve NLP tasks

https://github.com/rajcscw/nlp-gym

[NLPGym](https://arxiv.org/pdf/2011.08272v1.pdf) provides interactive environments for standard NLP tasks such as sequence tagging, question answering, and sequence classification. Users can easily customize the tasks with their own datasets, observations, featurizers and reward functions.

### Obstacle Tower

https://github.com/Unity-Technologies/obstacle-tower-env

3D procedurally generated tower where you have to climb to the highest level possible

### openmodelica-microgrid-gym

https://github.com/upb-lea/openmodelica-microgrid-gym

The OpenModelica Microgrid Gym (OMG) package is a software toolbox for the simulation and control optimization of microgrids based on energy conversion by power electronic converters.

### osim-rl

https://github.com/stanfordnmbl/osim-rl

Musculoskeletal Models in OpenSim. A human musculoskeletal model and a physics-based simulation environment where you can synthesize physically and physiologically accurate motion. One of the environments built in this framework is a competition environment for a NIPS 2017 challenge.

### PGE: Parallel Game Engine

https://github.com/222464/PGE

PGE is a FOSS 3D engine for AI simulations, and can interoperate with the Gym. Contains environments with modern 3D graphics, and uses Bullet for physics.

### QASGym

https://github.com/qdevpsi3/quantum-arch-search

This a list of environments for quantum architecture search following the description in [Quantum Architecture Search via Deep Reinforcement Learning](https://arxiv.org/abs/2104.07715). The agent design the quantum circuit by taking actions in the environment. Each action corresponds to a gate applied on some wires. The goal is to build a circuit U such that generates the target n-qubit quantum state that belongs to the environment and hidden from the agent. The circuits are built using [Google QuantumAI Cirq](https://quantumai.google/cirq).

### safe-control-gym

https://github.com/utiasDSL/safe-control-gym

PyBullet based CartPole and Quadrotor environments—with [CasADi](https://web.casadi.org) (symbolic) *a priori* dynamics and constraints—for learning-based control and model-based reinforcement learning.

### VirtualTaobao

https://github.com/eyounx/VirtualTaobao/

An environment for online recommendation, where customers are learned from Taobao.com, one of the world's largest e-commerce platform.
