import gym
import time
import numpy
import random
import pandas

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)


env = gym.make('GazeboMazeTurtlebotLidar-v0')


for x in range(3000):
    done = False
    env.reset()

    render_skip = 4 #Skip first X episodes.
    render_interval = 5 #Show render Every Y episodes.
    render_episodes = 2 #Show Z episodes every rendering.

    print "Episode: "+str(x)
    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        print "starting render"
        env.render()
    elif ((x-render_episodes)%(render_interval) == 0) and (x != 0) and (x > render_skip):
        print "closing render"
        env.render(close=True)


    for i in range(200):

        #action = env.action_space.sample() #not implemented
        #observation, reward, done, info = env.step(action)

        print "Ep: "+str(x)+" Ev:"+str(i)

        env.step(1)


        #Must change
        if (i%10 == 0) and (i != 0):
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
