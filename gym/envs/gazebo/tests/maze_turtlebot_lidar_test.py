import gym
import time

env = gym.make('GazeboMazeTurtlebotLidar-v0')


for x in range(3000):
    done = False
    env.reset()

    #Show render every 200 steps for 50 iterations
    print "Iteration: "+str(x)
    if (x%100 == 0) and (x != 0):
        print "starting render"
        env.render()
    elif (x%50 == 0) and (x != 0) and (x != 50):
        print "closing render"
        env.render(close=True)


    for i in range(500):

        #action = env.action_space.sample() #not implemented
        #observation, reward, done, info = env.step(action)

        env.step(1)

        if (i%25 == 0) and (i != 0):
            done = True
        if done:
            break 

env.close()


'''
print "Render starting"
env.render()

time.sleep(1)
env.reset()
print "reset simulation"

time.sleep(1)
print "Render starting"
env.render(close=True)


env.close()'''
