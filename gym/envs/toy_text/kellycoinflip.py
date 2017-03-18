import gym
from gym import spaces
from gym.utils import seeding

def flip(edge, np_random):
    return np_random.uniform() < edge

class KellyCoinflipEnv(gym.Env):
    """The Kelly coinflip game is a simple gambling introduced by Haghani & Dewey 2016's 'Rational Decision-Making Under Uncertainty: Observed Betting Patterns on a Biased Coin' (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2856963), to test human decision-making in a setting like that of the stock market: positive expected value but highly stochastic; they found many subjects performed badly, often going broke, even though optimal play would reach the maximum with ~95% probability. In the coinflip game, the player starts with $25.00 to gamble over 300 rounds; each round, they can bet anywhere up to their net worth (in penny increments), and then a coin is flipped; with P=0.6, the player wins twice what they bet, otherwise, they lose it. $250 is the maximum players are allowed to have. At the end of the 300 rounds, they keep whatever they have. The human subjects earned an average of $91; a simple use of the Kelly criterion (https://en.wikipedia.org/wiki/Kelly_criterion), giving a strategy of betting 20% until the cap is hit, would earn $240; a decision tree analysis shows that optimal play earns $246 (https://www.gwern.net/Coin-flip). The game short-circuits when either wealth = $0 (since one can never recover) or wealth = cap (trivial optimal play: one simply bets nothing thereafter). In this implementation, we default to the paper settings of $25, 60% odds, wealth cap of $250, and 300 rounds. To specify the action space in advance, we multiply the wealth cap (in dollars) by 100 (to allow for all penny bets); should one attempt to bet more money than one has, it is rounded down to one's net worth. (Alternately, a mistaken bet could end the episode immediately; it's not clear to me which version would be better.) TODO: Bayesian POMDP """
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
