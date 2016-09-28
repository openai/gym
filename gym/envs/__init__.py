from gym.envs.registration import registry, register, make, spec

# Algorithmic
# ----------------------------------------

register(
    id='Copy-v0',
    entry_point='gym.envs.algorithmic:CopyEnv',
    timestep_limit=200,
    reward_threshold=25.0,
)

register(
    id='RepeatCopy-v0',
    entry_point='gym.envs.algorithmic:RepeatCopyEnv',
    timestep_limit=200,
    reward_threshold=75.0,
)

register(
    id='ReversedAddition-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 2},
    timestep_limit=200,
    reward_threshold=25.0,
)

register(
    id='ReversedAddition3-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 3},
    timestep_limit=200,
    reward_threshold=25.0,
)

register(
    id='DuplicatedInput-v0',
    entry_point='gym.envs.algorithmic:DuplicatedInputEnv',
    timestep_limit=200,
    reward_threshold=9.0,
)

register(
    id='Reverse-v0',
    entry_point='gym.envs.algorithmic:ReverseEnv',
    timestep_limit=200,
    reward_threshold=25.0,
)

# Classic
# ----------------------------------------

register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    timestep_limit=200,
    reward_threshold=195.0,
)

register(
    id='CartPole-v1',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    timestep_limit=500,
    reward_threshold=475.0,
)

register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    timestep_limit=200,
    reward_threshold=-110.0,
)

register(
    id='MountainCarContinuous-v0',
    entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
    timestep_limit=999,
    reward_threshold=90.0,
)

register(
    id='Pendulum-v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    timestep_limit=200,
)

register(
    id='Acrobot-v1',
    entry_point='gym.envs.classic_control:AcrobotEnv',
    timestep_limit=500,
)

# Box2d
# ----------------------------------------

register(
    id='LunarLander-v2',
    entry_point='gym.envs.box2d:LunarLander',
    timestep_limit=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuous-v2',
    entry_point='gym.envs.box2d:LunarLanderContinuous',
    timestep_limit=1000,
    reward_threshold=200,
)

register(
    id='BipedalWalker-v2',
    entry_point='gym.envs.box2d:BipedalWalker',
    timestep_limit=1600,
    reward_threshold=300,
)

register(
    id='BipedalWalkerHardcore-v2',
    entry_point='gym.envs.box2d:BipedalWalkerHardcore',
    timestep_limit=2000,
    reward_threshold=300,
)

register(
    id='CarRacing-v0',
    entry_point='gym.envs.box2d:CarRacing',
    timestep_limit=1000,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id='Blackjack-v0',
    entry_point='gym.envs.toy_text:BlackjackEnv',
)

register(
    id='FrozenLake-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4'},
    timestep_limit=100,
    reward_threshold=0.78, # optimum = .8196
)

register(
    id='FrozenLake8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8'},
    timestep_limit=200,
    reward_threshold=0.99, # optimum = 1
)

register(
    id='NChain-v0',
    entry_point='gym.envs.toy_text:NChainEnv',
    timestep_limit=1000,
)

register(
    id='Roulette-v0',
    entry_point='gym.envs.toy_text:RouletteEnv',
    timestep_limit=100,
)

register(
    id='Taxi-v1',
    entry_point='gym.envs.toy_text.taxi:TaxiEnv',
    timestep_limit=200,
    reward_threshold=9.7, # optimum = 10.2
)

register(
    id='GuessingGame-v0',
    entry_point='gym.envs.toy_text.guessing_game:GuessingGame',
    timestep_limit=200,
)

register(
    id='HotterColder-v0',
    entry_point='gym.envs.toy_text.hotter_colder:HotterColder',
    timestep_limit=200,
)

# Mujoco
# ----------------------------------------

# 2D

register(
    id='Reacher-v1',
    entry_point='gym.envs.mujoco:ReacherEnv',
    timestep_limit=50,
    reward_threshold=-3.75,
)

register(
    id='InvertedPendulum-v1',
    entry_point='gym.envs.mujoco:InvertedPendulumEnv',
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulum-v1',
    entry_point='gym.envs.mujoco:InvertedDoublePendulumEnv',
    reward_threshold=9100.0,
)

register(
    id='HalfCheetah-v1',
    entry_point='gym.envs.mujoco:HalfCheetahEnv',
    reward_threshold=4800.0,
)

register(
    id='Hopper-v1',
    entry_point='gym.envs.mujoco:HopperEnv',
    reward_threshold=3800.0,
)

register(
    id='Swimmer-v1',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    reward_threshold=360.0,
)

register(
    id='Walker2d-v1',
    entry_point='gym.envs.mujoco:Walker2dEnv',
)

register(
    id='Ant-v1',
    entry_point='gym.envs.mujoco:AntEnv',
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v1',
    entry_point='gym.envs.mujoco:HumanoidEnv',
)
register(
    id='HumanoidStandup-v1',
    entry_point='gym.envs.mujoco:HumanoidStandupEnv',
)

# Atari
# ----------------------------------------

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True

        register(
            id='{}-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
            timestep_limit=10000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}-v3'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type},
            timestep_limit=100000,
            nondeterministic=nondeterministic,
        )

        # Standard Deterministic (as in the original DeepMind paper)
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        register(
            id='{}Deterministic-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip, 'repeat_action_probability': 0.25},
            timestep_limit=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}Deterministic-v3'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip},
            timestep_limit=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}NoFrameskip-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
            timestep_limit=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

        # No frameskip. (Atari has no entropy source, so these are
        # deterministic environments.)
        register(
            id='{}NoFrameskip-v3'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            timestep_limit=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

# Board games
# ----------------------------------------

register(
    id='Go9x9-v0',
    entry_point='gym.envs.board_game:GoEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'pachi:uct:_2400',
        'observation_type': 'image3c',
        'illegal_move_mode': 'lose',
        'board_size': 9,
    },
    # The pachi player seems not to be determistic given a fixed seed.
    # (Reproduce by running 'import gym; h = gym.make('Go9x9-v0'); h.seed(1); h.reset(); h.step(15); h.step(16); h.step(17)' a few times.)
    #
    # This is probably due to a computation time limit.
    nondeterministic=True,
)

register(
    id='Go19x19-v0',
    entry_point='gym.envs.board_game:GoEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'pachi:uct:_2400',
        'observation_type': 'image3c',
        'illegal_move_mode': 'lose',
        'board_size': 19,
    },
    nondeterministic=True,
)

register(
    id='Hex9x9-v0',
    entry_point='gym.envs.board_game:HexEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 9,
    },
)

# Debugging
# ----------------------------------------

register(
    id='OneRoundDeterministicReward-v0',
    entry_point='gym.envs.debugging:OneRoundDeterministicRewardEnv',
    local_only=True
)

register(
    id='TwoRoundDeterministicReward-v0',
    entry_point='gym.envs.debugging:TwoRoundDeterministicRewardEnv',
    local_only=True
)

register(
    id='OneRoundNondeterministicReward-v0',
    entry_point='gym.envs.debugging:OneRoundNondeterministicRewardEnv',
    local_only=True
)

register(
    id='TwoRoundNondeterministicReward-v0',
    entry_point='gym.envs.debugging:TwoRoundNondeterministicRewardEnv',
    local_only=True,
)

# Parameter tuning
# ----------------------------------------
register(
    id='ConvergenceControl-v0',
    entry_point='gym.envs.parameter_tuning:ConvergenceControl',
)

register(
    id='CNNClassifierTraining-v0',
    entry_point='gym.envs.parameter_tuning:CNNClassifierTraining',
)

# Safety
# ----------------------------------------

# interpretability envs
register(
    id='PredictActionsCartpole-v0',
    entry_point='gym.envs.safety:PredictActionsCartpoleEnv',
    timestep_limit=200,
)

register(
    id='PredictObsCartpole-v0',
    entry_point='gym.envs.safety:PredictObsCartpoleEnv',
    timestep_limit=200,
)

# semi_supervised envs
    # probably the easiest:
register(
    id='SemisuperPendulumNoise-v0',
    entry_point='gym.envs.safety:SemisuperPendulumNoiseEnv',
    timestep_limit=200,
)
    # somewhat harder because of higher variance:
register(
    id='SemisuperPendulumRandom-v0',
    entry_point='gym.envs.safety:SemisuperPendulumRandomEnv',
    timestep_limit=200,
)
    # probably the hardest because you only get a constant number of rewards in total:
register(
    id='SemisuperPendulumDecay-v0',
    entry_point='gym.envs.safety:SemisuperPendulumDecayEnv',
    timestep_limit=200,
)

# off_switch envs
register(
    id='OffSwitchCartpole-v0',
    entry_point='gym.envs.safety:OffSwitchCartpoleEnv',
    timestep_limit=200,
)

register(
    id='OffSwitchCartpoleProb-v0',
    entry_point='gym.envs.safety:OffSwitchCartpoleProbEnv',
    timestep_limit=200,
)
