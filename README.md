# OpenAI Gym 

This is a fork of [OpenAI gym](https://github.com/openai/gym) that Robot Locomotion Group uses for custom environments. These environments will include:
- Gym environments wrapped around [Drake](https://drake.mit.edu/)
- Modifications to existing [gym](https://github.com/openai/gym) environments where observations are directly pixels.
- Custom environments written in pymunk.

Some of these environments will be exported to a pip package once they are mature.

# List of New Environments

- Carrot pushing environment. `gym.make("Carrot-v0")`
- Pendulum environment with pixel output. `gym.make("PendulumPixel-v0")`

# Setup 

This fork can be setup in the following way.

```
git clone git@github.com:RobotLocomotion/gym.git
cd gym
pip install -e .
``` 

If you are working on this repo then add the following lines:
```
cd gym 
git remote set-url origin git@github.com:RobotLocomotion/gym.git
git remote add upstream git@github.com:openai/gym.git
git remote set-url --push upstream no_push
```

# Workflow 

You can create your own environment on a local branch or your own fork, then PR to this repo once the environment is good enough.

# How to Add a New Environment

See OpenAI's instructions on [creating a new environment](https://github.com/openai/gym/blob/master/docs/creating-environments.md). 

To isolate our environments from OpenAI's original environments, let's keep the group environments in `gym/envs/robot_locomotion_group`. Generally, the procedure will involve the following:

1. Create a folder under `envs/robot_locomotion_group` with the new environment name, and develop your environment there that contains your own `gym.Env` class.
2. Register your new environment class under `envs/robot_locomotion_group/__init__.py`.
3. Register your new environment name under `envs/__init__.py`. You will see a section dedicated to Robot Locomotion Grouop. 
4. Now you can use your environment from any location with `gym.make("my_awesome_new_environment-v0")`

# Dependencies 
- `pyglet`
- `pymunk`
- `drake`
