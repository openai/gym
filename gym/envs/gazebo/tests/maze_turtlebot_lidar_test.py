import gym
import time

env = gym.make('GazeboMazeTurtlebotLidar-v0')

time.sleep(5)

print "Render starting"
env.render()

time.sleep(1)
env.reset()
print "reset simulation"

time.sleep(1)
print "Render starting"
env.render(close=True)


env.close()
