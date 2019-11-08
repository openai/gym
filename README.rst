**Status:** Maintenance (expect bug fixes and minor updates)

OpenAI Gym
**********

**OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms.** This is the ``gym`` open-source library, which gives you access to a standardized set of environments.

.. image:: https://travis-ci.org/openai/gym.svg?branch=master
    :target: https://travis-ci.org/openai/gym

`See What's New section below <#what-s-new>`_

``gym`` makes no assumptions about the structure of your agent, and is compatible with any numerical computation library, such as TensorFlow or Theano. You can use it from Python code, and soon from other languages.

If you're not sure where to start, we recommend beginning with the
`docs <https://gym.openai.com/docs>`_ on our site. See also the `FAQ <https://github.com/openai/gym/wiki/FAQ>`_.

A whitepaper for OpenAI Gym is available at http://arxiv.org/abs/1606.01540, and here's a BibTeX entry that you can use to cite it in a publication::

  @misc{1606.01540,
    Author = {Greg Brockman and Vicki Cheung and Ludwig Pettersson and Jonas Schneider and John Schulman and Jie Tang and Wojciech Zaremba},
    Title = {OpenAI Gym},
    Year = {2016},
    Eprint = {arXiv:1606.01540},
  }

.. contents:: **Contents of this document**
   :depth: 2

Basics
======

There are two basic concepts in reinforcement learning: the
environment (namely, the outside world) and the agent (namely, the
algorithm you are writing). The agent sends `actions` to the
environment, and the environment replies with `observations` and
`rewards` (that is, a score).

The core `gym` interface is `Env <https://github.com/openai/gym/blob/master/gym/core.py>`_, which is
the unified environment interface. There is no interface for agents;
that part is left to you. The following are the ``Env`` methods you
should know:

- `reset(self)`: Reset the environment's state. Returns `observation`.
- `step(self, action)`: Step the environment by one timestep. Returns `observation`, `reward`, `done`, `info`.
- `render(self, mode='human')`: Render one frame of the environment. The default mode will do something human friendly, such as pop up a window. 

Supported systems
-----------------

We currently support Linux and OS X running Python 2.7 or 3.5 -- 3.7. 
Windows support is experimental - algorithmic, toy_text, classic_control and atari *should* work on Windows (see next section for installation instructions); nevertheless, proceed at your own risk.

Installation
============

You can perform a minimal install of ``gym`` with:

.. code:: shell

    git clone https://github.com/openai/gym.git
    cd gym
    pip install -e .

If you prefer, you can do a minimal install of the packaged version directly from PyPI:

.. code:: shell

    pip install gym

You'll be able to run a few environments right away:

- algorithmic
- toy_text
- classic_control (you'll need ``pyglet`` to render though)

We recommend playing with those environments at first, and then later
installing the dependencies for the remaining environments.

Installing everything
---------------------

To install the full set of environments, you'll need to have some system
packages installed. We'll build out the list here over time; please let us know
what you end up installing on your platform. Also, take a look at the docker files (py.Dockerfile) to
see the composition of our CI-tested images.

On Ubuntu 16.04 and 18.04:

.. code:: shell
    apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev


MuJoCo has a proprietary dependency we can't set up for you. Follow
the
`instructions <https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key>`_
in the ``mujoco-py`` package for help.  As an alternative to ``mujoco-py``, consider `PyBullet <https://github.com/openai/gym/blob/master/docs/environments.md#pybullet-robotics-environments>`_ which uses the open source Bullet physics engine and has no license requirement.

Once you're ready to install everything, run ``pip install -e '.[all]'`` (or ``pip install 'gym[all]'``).

Pip version
-----------

To run ``pip install -e '.[all]'``, you'll need a semi-recent pip.
Please make sure your pip is at least at version ``1.5.0``. You can
upgrade using the following: ``pip install --ignore-installed
pip``. Alternatively, you can open `setup.py
<https://github.com/openai/gym/blob/master/setup.py>`_ and
install the dependencies by hand.

Rendering on a server
---------------------

If you're trying to render video on a server, you'll need to connect a
fake display. The easiest way to do this is by running under
``xvfb-run`` (on Ubuntu, install the ``xvfb`` package):

.. code:: shell

     xvfb-run -s "-screen 0 1400x900x24" bash

Installing dependencies for specific environments
-------------------------------------------------

If you'd like to install the dependencies for only specific
environments, see `setup.py
<https://github.com/openai/gym/blob/master/setup.py>`_. We
maintain the lists of dependencies on a per-environment group basis.

Environments
============

See `List of Environments <docs/environments.md>`_ and the `gym site <http://gym.openai.com/envs/>`_.

For information on creating your own environments, see `Creating your own Environments <docs/creating-environments.md>`_.

Examples
========

See the ``examples`` directory.

- Run `examples/agents/random_agent.py <https://github.com/openai/gym/blob/master/examples/agents/random_agent.py>`_ to run a simple random agent.
- Run `examples/agents/cem.py <https://github.com/openai/gym/blob/master/examples/agents/cem.py>`_ to run an actual learning agent (using the cross-entropy method).
- Run `examples/scripts/list_envs <https://github.com/openai/gym/blob/master/examples/scripts/list_envs>`_ to generate a list of all environments.

Testing
=======

We are using `pytest <http://doc.pytest.org>`_ for tests. You can run them via:

.. code:: shell

    pytest


.. _See What's New section below:

What's new
==========
- 2019-11-08 (v0.15.4)
    + Added multiple env wrappers (thanks @zuoxingdong and @hartikainen!)
    - Removed mujoco >= 2.0 support due to lack of tests

- 2019-10-09 (v0.15.3)
    + VectorEnv modifications - unified the VectorEnv api (added reset_async, reset_wait, step_async, step_wait methods to SyncVectorEnv); more flexibility in AsyncVectorEnv workers

- 2019-08-23 (v0.15.2)
    + More Wrappers - AtariPreprocessing, FrameStack, GrayScaleObservation, FilterObservation,  FlattenDictObservationsWrapper, PixelObservationWrapper, TransformReward (thanks @zuoxingdong, @hartikainen)
    + Remove rgb_rendering_tracking logic from mujoco environments (default behavior stays the same for the -v3 environments, rgb rendering returns a view from tracking camera)
    + Velocity goal constraint for MountainCar (thanks @abhinavsagar)
    + Taxi-v2 -> Taxi-v3 (add missing wall in the map to replicate env as describe in the original paper, thanks @kobotics)
    
- 2019-07-26 (v0.14.0)
    + Wrapper cleanup
    + Spec-related bug fixes
    + VectorEnv fixes

- 2019-06-21 (v0.13.1)
    + Bug fix for ALE 0.6 difficulty modes
    + Use narrow range for pyglet versions

- 2019-06-21 (v0.13.0)
    + Upgrade to ALE 0.6 (atari-py 0.2.0) (thanks @JesseFarebro!)

- 2019-06-21 (v0.12.6)
    + Added vectorized environments (thanks @tristandeleu!). Vectorized environment runs multiple copies of an environment in parallel. To create a vectorized version of an environment, use `gym.vector.make(env_id, num_envs, **kwargs)`, for instance, `gym.vector.make('Pong-v4',16)`.

- 2019-05-28 (v0.12.5)
    + fixed Fetch-slide environment to be solvable.

- 2019-05-24 (v0.12.4)
    + remove pyopengl dependency and use more narrow atari-py and box2d-py versions

- 2019-03-25 (v0.12.1)
    + rgb rendering in MuJoCo locomotion `-v3` environments now comes from tracking camera (so that agent does not run away from the field of view). The old behaviour can be restored by passing rgb_rendering_tracking=False kwarg. Also, a potentially breaking change!!! Wrapper class now forwards methods and attributes to wrapped env.

- 2019-02-26 (v0.12.0)
    + release mujoco environments v3 with support for gym.make kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale` etc

- 2019-02-06 (v0.11.0)
    + remove gym.spaces.np_random common PRNG; use per-instance PRNG instead.
    + support for kwargs in gym.make
    + lots of bugfixes

- 2018-02-28: Release of a set of new robotics environments.
- 2018-01-25: Made some aesthetic improvements and removed unmaintained parts of gym. This may seem like a downgrade in functionality, but it is actually a long-needed cleanup in preparation for some great new things that will be released in the next month.

    + Now your `Env` and `Wrapper` subclasses should define `step`, `reset`, `render`, `close`, `seed` rather than underscored method names.
    + Removed the `board_game`, `debugging`, `safety`, `parameter_tuning` environments since they're not being maintained by us at OpenAI. We encourage authors and users to create new repositories for these environments.
    + Changed `MultiDiscrete` action space to range from `[0, ..., n-1]` rather than `[a, ..., b-1]`.
    + No more `render(close=True)`, use env-specific methods to close the rendering.
    + Removed `scoreboard` directory, since site doesn't exist anymore.
    + Moved `gym/monitoring` to `gym/wrappers/monitoring`
    + Add `dtype` to `Space`.
    + Not using python's built-in module anymore, using `gym.logger`

- 2018-01-24: All continuous control environments now use mujoco_py >= 1.50.
  Versions have been updated accordingly to -v2, e.g. HalfCheetah-v2. Performance
  should be similar (see https://github.com/openai/gym/pull/834) but there are likely
  some differences due to changes in MuJoCo.
- 2017-06-16: Make env.spec into a property to fix a bug that occurs
  when you try to print out an unregistered Env.
- 2017-05-13: BACKWARDS INCOMPATIBILITY: The Atari environments are now at
  *v4*. To keep using the old v3 environments, keep gym <= 0.8.2 and atari-py
  <= 0.0.21. Note that the v4 environments will not give identical results to
  existing v3 results, although differences are minor. The v4 environments
  incorporate the latest Arcade Learning Environment (ALE), including several
  ROM fixes, and now handle loading and saving of the emulator state. While
  seeds still ensure determinism, the effect of any given seed is not preserved
  across this upgrade because the random number generator in ALE has changed.
  The `*NoFrameSkip-v4` environments should be considered the canonical Atari
  environments from now on.
- 2017-03-05: BACKWARDS INCOMPATIBILITY: The `configure` method has been removed
  from `Env`. `configure` was not used by `gym`, but was used by some dependent
  libraries including `universe`. These libraries will migrate away from the
  configure method by using wrappers instead. This change is on master and will be released with 0.8.0.
- 2016-12-27: BACKWARDS INCOMPATIBILITY: The gym monitor is now a
  wrapper. Rather than starting monitoring as
  `env.monitor.start(directory)`, envs are now wrapped as follows:
  `env = wrappers.Monitor(env, directory)`. This change is on master
  and will be released with 0.7.0.
- 2016-11-1: Several experimental changes to how a running monitor interacts
  with environments. The monitor will now raise an error if reset() is called
  when the env has not returned done=True. The monitor will only record complete
  episodes where done=True. Finally, the monitor no longer calls seed() on the
  underlying env, nor does it record or upload seed information.
- 2016-10-31: We're experimentally expanding the environment ID format
  to include an optional username.
- 2016-09-21: Switch the Gym automated logger setup to configure the
  root logger rather than just the 'gym' logger.
- 2016-08-17: Calling `close` on an env will also close the monitor
  and any rendering windows.
- 2016-08-17: The monitor will no longer write manifest files in
  real-time, unless `write_upon_reset=True` is passed.
- 2016-05-28: For controlled reproducibility, envs now support seeding
  (cf #91 and #135). The monitor records which seeds are used. We will
  soon add seed information to the display on the scoreboard.
