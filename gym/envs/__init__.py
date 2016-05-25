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
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    timestep_limit=200,
    reward_threshold=-110.0,
)

register(
    id='Pendulum-v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    timestep_limit=200,
)

register(
    id='Acrobot-v0',
    entry_point='gym.envs.classic_control:AcrobotEnv',
    timestep_limit=200,
    reward_threshold=-100
)

# Box2d
# ----------------------------------------

register(
    id='LunarLander-v1',
    entry_point='gym.envs.box2d:LunarLander',
    timestep_limit=1000,
    reward_threshold=200,
)

register(
    id='BipedalWalker-v1',
    entry_point='gym.envs.box2d:BipedalWalker',
    timestep_limit=1600,
    reward_threshold=300,
)

register(
    id='BipedalWalkerHardcore-v1',
    entry_point='gym.envs.box2d:BipedalWalkerHardcore',
    timestep_limit=2000,
    reward_threshold=300,
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

# Mujoco
# ----------------------------------------

# 2D

register(
    id='Reacher-v1',
    entry_point='gym.envs.mujoco:ReacherEnv',
    timestep_limit=50
)

register(
    id='InvertedPendulum-v1',
    entry_point='gym.envs.mujoco:InvertedPendulumEnv',
)

register(
    id='InvertedDoublePendulum-v1',
    entry_point='gym.envs.mujoco:InvertedDoublePendulumEnv',
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
        register(
            id='{}-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type},
            timestep_limit=10000,
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

# Doom
# ----------------------------------------

register(
    id='DoomBasic-v0',
    entry_point='gym.envs.doom:DoomBasicEnv',
)

register(
    id='DoomCorridor-v0',
    entry_point='gym.envs.doom:DoomCorridorEnv',
)

register(
    id='DoomDefendCenter-v0',
    entry_point='gym.envs.doom:DoomDefendCenterEnv',
)

register(
    id='DoomDefendLine-v0',
    entry_point='gym.envs.doom:DoomDefendLineEnv',
)

register(
    id='DoomHealthGathering-v0',
    entry_point='gym.envs.doom:DoomHealthGatheringEnv',
)

register(
    id='DoomMyWayHome-v0',
    entry_point='gym.envs.doom:DoomMyWayHomeEnv',
)

register(
    id='DoomPredictPosition-v0',
    entry_point='gym.envs.doom:DoomPredictPositionEnv',
)

register(
    id='DoomTakeCover-v0',
    entry_point='gym.envs.doom:DoomTakeCoverEnv',
)

register(
    id='DoomDeathmatch-v0',
    entry_point='gym.envs.doom:DoomDeathmatchEnv',
)
