import random
import os

class QLearn:
    #def __init__(self, actions, epsilon, alpha, gamma):
    def __init__(self, actions, epsilon, alpha, gamma,epsilon_decay_rate):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        self.epsilon_decay_rate = epsilon_decay_rate #####


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
        self.epsilon *= self.epsilon_decay_rate ######
        # print ("epsilon", self.epsilon) #########

        if random.random() < self.epsilon:
            # print("RANDOM") ######
            action = random.choice(self.actions)######
            i = self.actions.index(action)
            maxQ = q[i]

            #minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            ##print("q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))", q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions)))#####
            ##q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            ##maxQ = max(q)

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

    #def learn(self, state1, action1, reward, state2):
        #maxqnew = max([self.getQ(state2, a) for a in self.actions])
        #print("maxqnew", maxqnew)
        #self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    def learn(self, state1, action1, reward, state2, save_model_with_prefix, it):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        # print("maxqnew", maxqnew)
        # for a in self.actions:
        #     print("q_value", self.getQ(state2, a))
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
