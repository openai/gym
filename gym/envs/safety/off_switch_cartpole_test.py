from gym.envs.safety.off_switch_cartpole import OffSwitchCartpoleEnv

env = OffSwitchCartpoleEnv()
for i in xrange(100):
    print env.step(1)
