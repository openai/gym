# DartEnv
Openai Gym with Dart support

### About

**DartEnv** provides 3D multi-body simulation environments based on the <a href="https://github.com/openai/gym">**openai gym**</a> environment. The physics simulation is carried out by <a href="http://dartsim.github.io/">Dart</a> and <a href="http://pydart2.readthedocs.io/en/latest/">PyDart2</a>, which is a python binding for Dart.

### Requirements

You need to install these packages first:

<a href="http://dartsim.github.io/">Dart</a>

<a href="http://pydart2.readthedocs.io/en/latest/">PyDart2</a>

You can choose to install <a href="http://dartsim.github.io/">Dart</a> and <a href="http://pydart2.readthedocs.io/en/latest/">PyDart2</a> in a quick way:

    sudo apt-add-repository ppa:dartsim
    sudo apt-get update
    sudo apt-get install libdart6-all-dev

For python2:

    sudo apt install python-pip
    pip install numpy
    pip install PyOpenGL PyOpenGL_accelerate
    sudo apt-get install swig python-pyqt5 python-pyqt5.qtopengl
    sudo pip install pydart2

For python3:

    sudo apt install python3-pip
    pip3 install numpy
    pip3 install PyOpenGL PyOpenGL_accelerate
    sudo apt-get install swig python3-pyqt5 python3-pyqt5.qtopengl
    sudo pip3 install pydart2


### Install

The installation is the same as for <a href="https://github.com/openai/gym">**openai gym**</a>. To install, simply do 

For python2:

    git clone https://github.com/VincentYu68/dart-env.git
    cd dart-env
    pip install -e .[dart]

For python3:

    git clone https://github.com/VincentYu68/dart-env.git
    cd dart-env
    pip3 install -e .[dart]

Please find the detailed installation instruction of using <a href="https://github.com/openai/rllab">Rllab</a> to learn DartEnv in the <a href="https://github.com/VincentYu68/dart-env/wiki">wiki</a> page.


### Example

After installation, you can run DartEnv using the same API as openai gym. One example of running the dart version of the Hopper model is shown below:

    import gym
    env = gym.make('DartHopper-v1')
    observation = env.reset()
    for i in range(100):
        observation, reward, done, envinfo = env.step(env.action_space.sample())
        env.render()

