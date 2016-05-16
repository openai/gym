import os

from gym.scoreboard.client.resource import FileUpload, Evaluation
from gym.scoreboard.registration import registry, add_task, add_group

# Discover API key from the environment. (You should never have to
# change api_base / web_base.)
api_key = os.environ.get('OPENAI_GYM_API_KEY')
api_base = os.environ.get('OPENAI_GYM_API_BASE', 'https://gym-api.openai.com')
web_base = os.environ.get('OPENAI_GYM_WEB_BASE', 'https://gym.openai.com')

# The following controls how various tasks appear on the
# scoreboard. These registrations can differ from what's registered in
# this repository.

# groups

add_group(
    id='classic_control',
    name='Classic control',
    description='Classic control problems from the RL literature.'
)

add_group(
    id='algorithmic',
    name='Algorithmic',
    description='Learn to imitate computations.',
)

add_group(
    id='atari',
    name='Atari',
    description='Reach high scores in Atari 2600 games.',
)

add_group(
    id='board_game',
    name='Board games',
    description='Play classic board games against strong opponents.',
)

add_group(
    id='box2d',
    name='Box2D',
    description='Continuous control tasks in the Box2D simulator.',
)

add_group(
    id='mujoco',
    name='MuJoCo',
    description='Continuous control tasks, running in a fast physics simulator.'
)

add_group(
    id='toy_text',
    name='Toy text',
    description='Simple text environments to get you started.'
)

# classic control

add_task(
    id='CartPole-v0',
    group='classic_control',
    summary="Balance a pole on a cart.",
    description="""\
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The system is controlled by applying a force of +1 or -1 to the cart.
The pendulum starts upright, and the goal is to prevent it from falling over.
A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the
cart moves more than 2.4 units from the center.
""",
    background="""\
This environment corresponds to the version of the cart-pole problem described by
Barto, Sutton, and Anderson [Barto83]_.

.. [Barto83] AG Barto, RS Sutton and CW Anderson, "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem", IEEE Transactions on Systems, Man, and Cybernetics, 1983.
""",
)


add_task(
    id='Acrobot-v0',
    group='classic_control',
    summary="Swing up a two-link robot.",
    description="""\
The acrobot system includes two joints and two links, where the joint between the two links is actuated.
Initially, the links are hanging downwards, and the goal is to swing the end of the lower link
up to a given height.
""",
    background="""\
The acrobot was first described by Sutton [Sutton96]_. We are using the version
from `RLPy <https://rlpy.readthedocs.org/en/latest/>`__ [Geramiford15]_, which uses Runge-Kutta integration for better accuracy.

.. [Sutton96] R Sutton, "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding", NIPS 1996.
.. [Geramiford15] A Geramifard, C Dann, RH Klein, W Dabney, J How, "RLPy: A Value-Function-Based Reinforcement Learning Framework for Education and Research." JMLR, 2015.
""",
)

add_task(
    id='MountainCar-v0',
    group='classic_control',
    summary="Drive up a big hill.",
    description="""
A car is on a one-dimensional track,
positioned between two "mountains".
The goal is to drive up the mountain on the right; however, the car's engine is not
strong enough to scale the mountain in a single pass.
Therefore, the only way to succeed is to drive back and forth to build up momentum.
""",
    background="""\
This problem was first described by Andrew Moore in his PhD thesis [Moore90]_.

.. [Moore90] A Moore, Efficient Memory-Based Learning for Robot Control, PhD thesis, University of Cambridge, 1990.
""",
)

add_task(
    id='Pendulum-v0',
    group='classic_control',
    summary="Swing up a pendulum.",
    description="""
The inverted pendulum swingup problem is a classic problem in the control literature.
In this version of the problem, the pendulum starts in a random position, and the goal is to
swing it up so it stays upright.
"""
)

# algorithmic

add_task(
    id='Copy-v0',
    group='algorithmic',
    summary='Copy symbols from the input tape.',
    description="""
This task involves copying the symbols from the input tape to the output
tape. Although simple, the model still has to learn the correspondence
between input and output symbols, as well as executing the move right
action on the input tape.
""",
)

add_task(
    id='RepeatCopy-v0',
    group='algorithmic',
    summary='Copy symbols from the input tape multiple times.',
    description=r"""
A generic input is :math:`[mx_1 x_2 \ldots x_k]` and the desired output is :math:`[x_1 x_2 \ldots x_k x_k \ldots x_2 x_1 x_1 x_2 \ldots x_k x_1 x_2 \ldots x_k]`. Thus the goal is to copy the input, revert it and copy it again.
"""
)

add_task(
    id='DuplicatedInput-v0',
    group='algorithmic',
    summary='Copy and deduplicate data from the input tape.',
    description=r"""
The input tape has the form :math:`[x_1 x_1 x_1 x_2 x_2 x_2 \ldots
x_k x_k x_k]`, while the desired output is :math:`[x_1 x_2 \ldots x_k]`.
Thus each input symbol is replicated three times, so the model must emit
every third input symbol.
""",
)

add_task(
    id='ReversedAddition-v0',
    group='algorithmic',
    summary='Learn to add multi-digit numbers.',
    description="""
The goal is to add two multi-digit sequences, provided on an input
grid. The sequences are provided in two adjacent rows, with the right edges
aligned. The initial position of the read head is the last digit of the top number
(i.e. upper-right corner). The model has to: (i) memorize an addition table
for pairs of digits; (ii) learn how to move over the input grid and (iii) discover
the concept of a carry.
""",
)

add_task(
    id='ReversedAddition3-v0',
    group='algorithmic',
    summary='Learn to add three multi-digit numbers.',
    description="""
Same as the addition task, but now three numbers are
to be added. This is more challenging as the reward signal is less frequent (since
more correct actions must be completed before a correct output digit can be
produced). Also the carry now can take on three states (0, 1 and 2), compared
with two for the 2 number addition task.
""",
)

add_task(
    id='Reverse-v0',
    group='algorithmic',
    summary='Reverse the symbols on the input tape.',
    description="""
The goal is to reverse a sequence of symbols on the input tape. We provide
a special character :math:`r` to indicate the end of the sequence. The model
must learn to move right multiple times until it hits the :math:`r` symbol, then
move to the left, copying the symbols to the output tape.
""",
)

# board_game

add_task(
    id='Go9x9-v0',
    group='board_game',
    summary='The ancient game of Go, played on a 9x9 board.',
)

add_task(
    id='Go19x19-v0',
    group='board_game',
    summary='The ancient game of Go, played on a 19x19 board.',
)

add_task(
    id='Hex9x9-v0',
    group='board_game',
    summary='Hex played on a 9x9 board.',
)

# box2d

add_task(
    id='LunarLander-v1',
    group='box2d',
    experimental=True,
)

add_task(
    id='BipedalWalker-v1',
    group='box2d',
    experimental=True,
)

add_task(
    id='BipedalWalkerHardcore-v1',
    group='box2d',
    experimental=True,
)

# mujoco

add_task(
    id='InvertedPendulum-v1',
    summary="Balance a pole on a cart.",
    group='mujoco',
)

add_task(
    id='InvertedDoublePendulum-v1',
    summary="Balance a pole on a pole on a cart.",
    group='mujoco',
)

add_task(
    id='Reacher-v1',
    summary="Make a 2D robot reach to a randomly located target.",
    group='mujoco',
)

add_task(
    id='HalfCheetah-v1',
    summary="Make a 2D cheetah robot run.",
    group='mujoco',
)


add_task(
    id='Swimmer-v1',
    group='mujoco',
    summary="Make a 2D robot swim.",
    description="""
This task involves a 3-link swimming robot in a viscous fluid, where the goal is to make it
swim forward as fast as possible, by actuating the two joints.
The origins of task can be traced back to Remi Coulom's thesis [1]_.

.. [1] R Coulom. "Reinforcement Learning Using Neural Networks, with Applications to Motor Control". PhD thesis, Institut National Polytechnique de Grenoble, 2002.
"""
)

add_task(
    id='Hopper-v1',
    summary="Make a 2D robot hop.",
    group='mujoco',
    description="""\
Make a two-dimensional one-legged robot hop forward as fast as possible.
""",
    background="""\
The robot model is based on work by Erez, Tassa, and Todorov [Erez11]_.

.. [Erez11] T Erez, Y Tassa, E Todorov, "Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks", 2011.

""",
)

add_task(
    id='Walker2d-v1',
    summary="Make a 2D robot walk.",
    group='mujoco',
    description="""\
Make a two-dimensional bipedal robot walk forward as fast as possible.
""",
    background="""\
The robot model is based on work by Erez, Tassa, and Todorov [Erez11]_.

.. [Erez11] T Erez, Y Tassa, E Todorov, "Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks", 2011.

""",
)


add_task(
    id='Ant-v1',
    group='mujoco',
    summary="Make a 3D four-legged robot walk.",
    description ="""\
Make a four-legged creature walk forward as fast as possible.
""",
    background="""\
This task originally appeared in [Schulman15]_.

.. [Schulman15] J Schulman, P Moritz, S Levine, M Jordan, P Abbeel, "High-Dimensional Continuous Control Using  Generalized Advantage Estimation," ICLR, 2015.
""",
)

add_task(
    id='Humanoid-v1',
    group='mujoco',
    summary="Make a 3D two-legged robot walk.",
    description="""\
Make a three-dimensional bipedal robot walk forward as fast as possible, without falling over.
""",
    background="""\
The robot model was originally created by Tassa et al. [Tassa12]_.

.. [Tassa12] Y Tassa, T Erez, E Todorov, "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization".
""",
)

# toy text

add_task(
    id='FrozenLake-v0',
    group='toy_text',
    summary='Find a safe path across a grid of ice and water tiles.',
    description="""
The agent controls the movement of a character in a grid world. Some tiles
of the grid are walkable, and others lead to the agent falling into the water.
Additionally, the movement direction of the agent is uncertain and only partially
depends on the chosen direction.
The agent is rewarded for finding a walkable path to a goal tile.
""",
    background="""
Winter is here. You and your friends were tossing around a frisbee at the park
when you made a wild throw that left the frisbee out in the middle of the lake.
The water is mostly frozen, but there are a few holes where the ice has melted.
If you step into one of those holes, you'll fall into the freezing water.
At this time, there's an international frisbee shortage, so it's absolutely
imperative that you navigate across the lake and retrieve the disc.
However, the ice is slippery, so you won't always move in the direction you intend.

The surface is described using a grid like the following::

    SFFF       (S: starting point, safe)
    FHFH       (F: frozen surface, safe)
    FFFH       (H: hole, fall to your doom)
    HFFG       (G: goal, where the frisbee is located)

The episode ends when you reach the goal or fall in a hole.
You receive a reward of 1 if you reach the goal, and zero otherwise.
""",
)

add_task(
    id='FrozenLake8x8-v0',
    group='toy_text',
)

add_task(
    id='Taxi-v1',
    group='toy_text',
    summary='As a taxi driver, you need to pick up and drop off passengers as fast as possible.',
    description="""
This task was introduced in [Dietterich2000] to illustrate some issues in hierarchical reinforcement learning.
There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another.
You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty
for illegal pick-up and drop-off actions.

.. [Dietterich2000] T Erez, Y Tassa, E Todorov, "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition", 2011.
"""
)

add_task(
    id='Roulette-v0',
    group='toy_text',
    summary='Learn a winning strategy for playing roulette.',
    description="""
The agent plays 0-to-36 Roulette in a modified casino setting. For each spin,
the agent bets on a number. The agent receives a positive reward
iff the rolled number is not zero and its parity matches the agent's bet.
Additionally, the agent can choose to walk away from the table, ending the
episode.
""",
    background="""
The modification from classical Roulette is to reduce variance -- agents can
learn more quickly that the reward from betting on any number is uniformly
distributed. Additionally, rational agents should learn that the best long-term
move is not to play at all, but to walk away from the table.
""",
)

add_task(
    id='NChain-v0',
    group='toy_text',
    experimental=True,
)

add_task(
    id='Blackjack-v0',
    group='toy_text',
    experimental=True,
)

ram_desc = "In this environment, the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes."
image_desc = "In this environment, the observation is an RGB image of the screen, which is an array of shape (210, 160, 3)"

for id in sorted(['AirRaid-v0', 'AirRaid-ram-v0', 'Alien-v0', 'Alien-ram-v0', 'Amidar-v0', 'Amidar-ram-v0', 'Assault-v0', 'Assault-ram-v0', 'Asterix-v0', 'Asterix-ram-v0', 'Asteroids-v0', 'Asteroids-ram-v0', 'Atlantis-v0', 'Atlantis-ram-v0', 'BankHeist-v0', 'BankHeist-ram-v0', 'BattleZone-v0', 'BattleZone-ram-v0', 'BeamRider-v0', 'BeamRider-ram-v0', 'Berzerk-v0', 'Berzerk-ram-v0', 'Bowling-v0', 'Bowling-ram-v0', 'Boxing-v0', 'Boxing-ram-v0', 'Breakout-v0', 'Breakout-ram-v0', 'Carnival-v0', 'Carnival-ram-v0', 'Centipede-v0', 'Centipede-ram-v0', 'ChopperCommand-v0', 'ChopperCommand-ram-v0', 'CrazyClimber-v0', 'CrazyClimber-ram-v0', 'DemonAttack-v0', 'DemonAttack-ram-v0', 'DoubleDunk-v0', 'DoubleDunk-ram-v0', 'ElevatorAction-v0', 'ElevatorAction-ram-v0', 'Enduro-v0', 'Enduro-ram-v0', 'FishingDerby-v0', 'FishingDerby-ram-v0', 'Freeway-v0', 'Freeway-ram-v0', 'Frostbite-v0', 'Frostbite-ram-v0', 'Gopher-v0', 'Gopher-ram-v0', 'Gravitar-v0', 'Gravitar-ram-v0', 'IceHockey-v0', 'IceHockey-ram-v0', 'Jamesbond-v0', 'Jamesbond-ram-v0', 'JourneyEscape-v0', 'JourneyEscape-ram-v0', 'Kangaroo-v0', 'Kangaroo-ram-v0', 'Krull-v0', 'Krull-ram-v0', 'KungFuMaster-v0', 'KungFuMaster-ram-v0', 'MontezumaRevenge-v0', 'MontezumaRevenge-ram-v0', 'MsPacman-v0', 'MsPacman-ram-v0', 'NameThisGame-v0', 'NameThisGame-ram-v0', 'Phoenix-v0', 'Phoenix-ram-v0', 'Pitfall-v0', 'Pitfall-ram-v0', 'Pong-v0', 'Pong-ram-v0', 'Pooyan-v0', 'Pooyan-ram-v0', 'PrivateEye-v0', 'PrivateEye-ram-v0', 'Qbert-v0', 'Qbert-ram-v0', 'Riverraid-v0', 'Riverraid-ram-v0', 'RoadRunner-v0', 'RoadRunner-ram-v0', 'Robotank-v0', 'Robotank-ram-v0', 'Seaquest-v0', 'Seaquest-ram-v0', 'Skiing-v0', 'Skiing-ram-v0', 'Solaris-v0', 'Solaris-ram-v0', 'SpaceInvaders-v0', 'SpaceInvaders-ram-v0', 'StarGunner-v0', 'StarGunner-ram-v0', 'Tennis-v0', 'Tennis-ram-v0', 'TimePilot-v0', 'TimePilot-ram-v0', 'Tutankham-v0', 'Tutankham-ram-v0', 'UpNDown-v0', 'UpNDown-ram-v0', 'Venture-v0', 'Venture-ram-v0', 'VideoPinball-v0', 'VideoPinball-ram-v0', 'WizardOfWor-v0', 'WizardOfWor-ram-v0', 'YarsRevenge-v0', 'YarsRevenge-ram-v0', 'Zaxxon-v0', 'Zaxxon-ram-v0']):
    try:
        split = id.split("-")
        game = split[0]
        if len(split) == 2:
            ob_type = 'image'
        else:
            ob_type = 'ram'
    except ValueError as e:
        raise ValueError('{}: id={}'.format(e, id))
    ob_desc = ram_desc if ob_type == "ram" else image_desc
    add_task(
        id=id,
        group='atari',
        summary="Maximize score in the game %(game)s, with %(ob_type)s as input"%dict(game=game, ob_type="RAM" if ob_type=="ram" else "screen images"),
        description="""\
Maximize your score in the Atari 2600 game %(game)s.
%(ob_desc)s
Each action is repeatedly performed for a duration of :math:`k` frames,
where :math:`k` is uniformly sampled from :math:`\{2, 3, 4\}`.
"""%dict(game=game, ob_desc=ob_desc),
        background="""\
The game is simulated through the Arcade Learning Environment [ALE]_, which uses the Stella [Stella]_ Atari emulator.

.. [ALE] MG Bellemare, Y Naddaf, J Veness, and M Bowling. "The arcade learning environment: An evaluation platform for general agents." Journal of Artificial Intelligence Research (2012).
.. [Stella] Stella: A Multi-Platform Atari 2600 VCS emulator http://stella.sourceforge.net/
""",
    )

# Deprecated

# MuJoCo

add_task(
    id='InvertedPendulum-v0',
    summary="Balance a pole on a cart.",
    group='mujoco',
    deprecated=True,
)

add_task(
    id='InvertedDoublePendulum-v0',
    summary="Balance a pole on a pole on a cart.",
    group='mujoco',
    deprecated=True,
)

add_task(
    id='Reacher-v0',
    summary="Make a 2D robot reach to a randomly located target.",
    group='mujoco',
    deprecated=True,
)

add_task(
    id='HalfCheetah-v0',
    summary="Make a 2D cheetah robot run.",
    group='mujoco',
    deprecated=True,
)

add_task(
    id='Swimmer-v0',
    group='mujoco',
    summary="Make a 2D robot swim.",
    description="""
This task involves a 3-link swimming robot in a viscous fluid, where the goal is to make it
swim forward as fast as possible, by actuating the two joints.
The origins of task can be traced back to Remi Coulom's thesis [1]_.

.. [1] R Coulom. "Reinforcement Learning Using Neural Networks, with Applications to Motor Control". PhD thesis, Institut National Polytechnique de Grenoble, 2002.
    """,
    deprecated=True,
)

add_task(
    id='Hopper-v0',
    summary="Make a 2D robot hop.",
    group='mujoco',
    description="""\
Make a two-dimensional one-legged robot hop forward as fast as possible.
""",
    background="""\
The robot model is based on work by Erez, Tassa, and Todorov [Erez11]_.

.. [Erez11] T Erez, Y Tassa, E Todorov, "Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks", 2011.

""",
    deprecated=True,
)

add_task(
    id='Walker2d-v0',
    summary="Make a 2D robot walk.",
    group='mujoco',
    description="""\
Make a two-dimensional bipedal robot walk forward as fast as possible.
""",
    background="""\
The robot model is based on work by Erez, Tassa, and Todorov [Erez11]_.

.. [Erez11] T Erez, Y Tassa, E Todorov, "Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks", 2011.

""",
    deprecated=True,
)


add_task(
    id='Ant-v0',
    group='mujoco',
    summary="Make a 3D four-legged robot walk.",
    description ="""\
Make a four-legged creature walk forward as fast as possible.
""",
    background="""\
This task originally appeared in [Schulman15]_.

.. [Schulman15] J Schulman, P Moritz, S Levine, M Jordan, P Abbeel, "High-Dimensional Continuous Control Using  Generalized Advantage Estimation," ICLR, 2015.
""",
    deprecated=True,
)

add_task(
    id='Humanoid-v0',
    group='mujoco',
    summary="Make a 3D two-legged robot walk.",
    description="""\
Make a three-dimensional bipedal robot walk forward as fast as possible, without falling over.
""",
    background="""\
The robot model was originally created by Tassa et al. [Tassa12]_.

.. [Tassa12] Y Tassa, T Erez, E Todorov, "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization".
""",
    deprecated=True,
)

registry.finalize()
