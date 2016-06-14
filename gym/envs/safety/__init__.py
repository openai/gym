# interpretability envs
from gym.envs.safety.predict_actions_cartpole import PredictActionsCartpoleEnv
from gym.envs.safety.predict_obs_cartpole import PredictObsCartpoleEnv

# semi_supervised envs
    # probably the easiest:
from gym.envs.safety.semisuper_pendulum_noise import SemisuperPendulumNoiseEnv
    # somewhat harder because of higher variance:
from gym.envs.safety.semisuper_pendulum_random import SemisuperPendulumRandomEnv
    # probably the hardest because you only get a constant number of rewards in total:
from gym.envs.safety.semisuper_pendulum_decay import SemisuperPendulumDecayEnv

# off_switch envs
from gym.envs.safety.offswitch_cartpole import OffSwitchCartpoleEnv
