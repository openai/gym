# DartEnv
Openai Gym with Dart support

###About

**DartEnv** provides 3D multi-body simulation environments based on the <a href="https://github.com/openai/gym">**openai gym**</a> environment. The physics simulation is carried out by <a href="http://dartsim.github.io/">Dart</a> and <a href="http://pydart2.readthedocs.io/en/latest/">PyDart2</a>, which is a python binding for Dart.

###Requirements

You need to install these packages first:

<a href="http://dartsim.github.io/">Dart</a>

<a href="http://pydart2.readthedocs.io/en/latest/">PyDart2</a>

###Install

To facilitate installation, we have uploaded the entire project base including the original openai gym code. To install, simply do 


    git clone https://github.com/VincentYu68/DartEnv.git
    cd DartEnv
    pip install -e .


Please find the detailed installation instruction of using <a href="https://github.com/openai/rllab">Rllab</a> to learn DartEnv in the <a href="https://github.com/VincentYu68/DartEnv/wiki">wiki</a> page.


###Example

After installation, you can run DartEnv using the same API as openai gym. One example of running the dart version of the Hopper model is shown below:

    import gym
    env = gym.make('DartHopper-v1')
    observation = env.reset()
    for i in range(100):
        observation, reward, done, envinfo = env.step(env.action_space.sample())
        env.render()

