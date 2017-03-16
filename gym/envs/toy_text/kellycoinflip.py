import gym
from gym import spaces
from gym.utils import seeding

def flip(edge, np_random):
    return np_random.uniform() < edge

class KellyCoinflipEnv(gym.Env):
    """ """
    metadata = {'render.modes': ['human']}
    def __init__(self, initialWealth=25, edge=0.6, maxWealth=250, maxRounds=300):

        self.action_space = spaces.Discrete(maxWealth*100) # betting in penny increments
        self.observation_space = spaces.Tuple((
            spaces.Discrete(maxWealth*100+1), # (w,b)
            spaces.Discrete(maxRounds+1)))
        self.reward_range = (0, maxWealth)
        self.edge = edge
        self.wealth = initialWealth
        self.initialWealth = initialWealth
        self.maxRounds = maxRounds
        self.maxWealth = maxWealth
        self._seed()
        self._reset()
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def _step(self, action):
        action = action/100 # convert from pennies to dollars
        if action > self.wealth: # treat attempts to bet more than possess as == betting everything
          action = self.wealth
        if self.wealth <= 0:
            done = True
            reward = 0
        else:
          if self.rounds == 0:
            done = True
            reward = self.wealth
          else:
            self.rounds = self.rounds - 1
            done = False
            reward = 0
            coinflip = flip(self.edge, self.np_random)
            if coinflip:
              self.wealth = min(self.maxWealth, self.wealth + action)
            else:
                self.wealth = self.wealth - action
        return self._get_obs(), reward, done, {}
    def _get_obs(self):
        return (self.wealth, self.rounds)
    def _reset(self):
        self.rounds = self.maxRounds
        self.wealth = self.initialWealth
        return self._get_obs()
    def _render(self, mode='human', close=True):
        print("Current wealth: ", self.wealth, "; Rounds left: ", self.rounds)
