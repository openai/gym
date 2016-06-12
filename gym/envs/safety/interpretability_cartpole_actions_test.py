from gym.envs.safety.interpretability_cartpole_actions import InterpretabilityCartpoleActionsEnv

env = InterpretabilityCartpoleActionsEnv()
for i in xrange(100):
    print env.step([1, 0, 1, 0, 1, 0])
    print env.step([0, 1, 0, 1, 0, 1])
