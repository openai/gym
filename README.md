# An OpenAI gym extension for using Gazebo known as `gym-gazebo`

<!--[![alt tag](https://travis-ci.org/erlerobot/gym.svg?branch=master)](https://travis-ci.org/erlerobot/gym)-->

This work presents an extension of the initial OpenAI gym for robotics using ROS and Gazebo. A whitepaper about this work is available at https://arxiv.org/abs/1608.05742. Please use the following BibTex entry to cite our work:

	@misc{1608.05742,
		Author = {Iker Zamora and Nestor Gonzalez Lopez and Victor Mayoral Vilches and Alejandro Hernandez Cordero},
		Title = {Extending the OpenAI Gym for robotics: a toolkit for reinforcement learning using ROS and Gazebo},
		Year = {2016},
		Eprint = {arXiv:1608.05742},
	}

Visit [erlerobotics/gym](http://erlerobotics.com/docs/Simulation/Gym/) for more information and videos.

## Table of Contents
- [Environments](#environments)
- [Installation](#installation)
- [Usage](#usage)


## Environments
The following are some of the available gazebo environments:

| Name | Status | Description |
| ---- | ------ | ----------- |
| `GazeboCircuit2TurtlebotLidar-v0` | Maintained | A simple circuit with straight tracks and 90 degree turns. Highly discretized LIDAR readings are used to train the Turtlebot. Scripts implementing **Q-learning** and **Sarsa** can be found in the _examples_ folder. |
| `GazeboCircuit2TurtlebotLidarNn-v0` | | A simple circuit with straight tracks and 90 degree turns. A LIDAR is used to train the Turtlebot. Scripts implementing **DQN** can be found in the _examples_ folder.|
| `GazeboCircuit2cTurtlebotCameraNnEnv-v0` | | A simple circuit with straight tracks and 90 degree turns with high contrast colors between the floor and the walls. A camera is used to train the Turtlebot. Scripts implementing **DQN** using **CNN** can be found in the _examples_ folder. |


## Installation
Refer to [INSTALL.md](INSTALL.md)

## Usage

### Build and install gym-gazebo

In the root directory of the repository:

```bash
sudo pip install -e .
```

### Running an environment

- Load the environment variables corresponding to the robot you want to launch. E.g. to load the Turtlebot:

```bash
cd gym_gazebo/envs/installation
bash turtlebot_setup.bash
```

Note: all the setup scripts are available in `gym_gazebo/envs/installation`

- Run any of the examples available in `examples/`. E.g.:

```bash
cd examples/scripts_turtlebot
python circuit2_turtlebot_lidar_qlearn.py
```

### Display the simulation

To see what's going on in Gazebo during a simulation, simply run gazebo client:

```bash
gzclient
```

### Display reward plot

Display a graph showing the current reward history by running the following script:

```bash
cd examples/utilities
python display_plot.py
```

HINT: use `--help` flag for more options.

### Killing background processes

Sometimes, after ending or killing the simulation `gzserver` and `rosmaster` stay on the background, make sure you end them before starting new tests.

We recommend creating an alias to kill those processes.

```bash
echo "alias killgazebogym='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient'" >> ~/.bashrc
```
