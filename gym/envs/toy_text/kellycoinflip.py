import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import prng
# for Generalized Kelly coinflip game distributions:
from scipy.stats import genpareto
import numpy as np
import numpy.random

def flip(edge, np_random):
    return np_random.uniform() < edge

class KellyCoinflipEnv(gym.Env):
    """The Kelly coinflip game is a simple gambling introduced by Haghani & Dewey 2016's 'Rational Decision-Making Under Uncertainty: Observed Betting Patterns on a Biased Coin' (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2856963), to test human decision-making in a setting like that of the stock market: positive expected value but highly stochastic; they found many subjects performed badly, often going broke, even though optimal play would reach the maximum with ~95% probability. In the coinflip game, the player starts with $25.00 to gamble over 300 rounds; each round, they can bet anywhere up to their net worth (in penny increments), and then a coin is flipped; with P=0.6, the player wins twice what they bet, otherwise, they lose it. $250 is the maximum players are allowed to have. At the end of the 300 rounds, they keep whatever they have. The human subjects earned an average of $91; a simple use of the Kelly criterion (https://en.wikipedia.org/wiki/Kelly_criterion), giving a strategy of betting 20% until the cap is hit, would earn $240; a decision tree analysis shows that optimal play earns $246 (https://www.gwern.net/Coin-flip). The game short-circuits when either wealth = $0 (since one can never recover) or wealth = cap (trivial optimal play: one simply bets nothing thereafter). In this implementation, we default to the paper settings of $25, 60% odds, wealth cap of $250, and 300 rounds. To specify the action space in advance, we multiply the wealth cap (in dollars) by 100 (to allow for all penny bets); should one attempt to bet more money than one has, it is rounded down to one's net worth. (Alternately, a mistaken bet could end the episode immediately; it's not clear to me which version would be better.) For a harder version which randomizes the 3 key parameters, see the Generalized Kelly coinflip game."""
    metadata = {'render.modes': ['human']}
    def __init__(self, initialWealth=25.0, edge=0.6, maxWealth=250.0, maxRounds=300):

        self.action_space = spaces.Discrete(int(maxWealth*100)) # betting in penny increments
        self.observation_space = spaces.Tuple((
            spaces.Box(0, maxWealth, [1]), # (w,b)
            spaces.Discrete(maxRounds+1)))
        self.reward_range = (0, maxWealth)
        self.edge = edge
        self.wealth = initialWealth
        self.initialWealth = initialWealth
        self.maxRounds = maxRounds
        self.maxWealth = maxWealth
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = action/100.0 # convert from pennies to dollars
        if action > self.wealth: # treat attempts to bet more than possess as == betting everything
          action = self.wealth
        if self.wealth < 0.000001:
            done = True
            reward = 0.0
        else:
          if self.rounds == 0:
            done = True
            reward = self.wealth
          else:
            self.rounds = self.rounds - 1
            done = False
            reward = 0.0
            coinflip = flip(self.edge, self.np_random)
            if coinflip:
              self.wealth = min(self.maxWealth, self.wealth + action)
            else:
              self.wealth = self.wealth - action
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (np.array([self.wealth]), self.rounds)

    def reset(self):
        self.rounds = self.maxRounds
        self.wealth = self.initialWealth
        return self._get_obs()

    def render(self, mode='human'):
        print("Current wealth: ", self.wealth, "; Rounds left: ", self.rounds)

class KellyCoinflipGeneralizedEnv(gym.Env):
    """The Generalized Kelly coinflip game is an extension by ArthurB & Gwern Branwen which expands the Kelly coinflip game MDP into a POMDP, where the 3 key parameters (edge, maximum wealth, and number of rounds) are unknown random variables drawn from 3 distributions: a Beta(7,3) for the coinflip edge 0-1, a N(300,25) the total number of rounds, and a Pareto(5,200) for the wealth cap. These distributions are chosen to be conjugate & easily updatable, to allow for inference (other choices like the geometric for number of rounds wouldn't make observations informative), and to loosely reflect what a human might expect in the original Kelly coinflip game given that the number of rounds wasn't strictly fixed and they weren't told the wealth cap until they neared it. With these particular distributions, the entire history of the game can be summarized into a few sufficient statistics of rounds-elapsed/wins/losses/max-wealth-ever-reached, from which the Bayes-optimal decision can (in theory) be made; to avoid all agents having to tediously track those sufficient statistics manually in the same way, the observation space is augmented from wealth/rounds-left (rounds-left is deleted because it is a hidden variable) to current-wealth/rounds-elapsed/wins/losses/maximum-observed-wealth. The simple Kelly coinflip game can easily be solved by calculating decision trees, but the Generalized Kelly coinflip game may be intractable (although the analysis for the edge case alone suggests that the Bayes-optimal value may be very close to what one would calculate using a decision tree for any specific case), and represents a good challenge for RL agents."""
    metadata = {'render.modes': ['human']}
    def __init__(self, initialWealth=25.0, edgePriorAlpha=7, edgePriorBeta=3, maxWealthAlpha=5.0, maxWealthM=200.0, maxRoundsMean=300.0, maxRoundsSD=25.0, reseed=True):
        # store the hyperparameters for passing back into __init__() during resets so the same hyperparameters govern the next game's parameters, as the user expects: TODO: this is boilerplate, is there any more elegant way to do this?
        self.initialWealth=float(initialWealth)
        self.edgePriorAlpha=edgePriorAlpha
        self.edgePriorBeta=edgePriorBeta
        self.maxWealthAlpha=maxWealthAlpha
        self.maxWealthM=maxWealthM
        self.maxRoundsMean=maxRoundsMean
        self.maxRoundsSD=maxRoundsSD

        # draw this game's set of parameters:
        edge = prng.np_random.beta(edgePriorAlpha, edgePriorBeta)
        maxWealth = round(genpareto.rvs(maxWealthAlpha, maxWealthM, random_state=prng.np_random))
        maxRounds = int(round(prng.np_random.normal(maxRoundsMean, maxRoundsSD)))

        # add an additional global variable which is the sufficient statistic for the Pareto distribution on wealth cap;
        # alpha doesn't update, but x_m does, and simply is the highest wealth count we've seen to date:
        self.maxEverWealth = float(self.initialWealth)
        # for the coinflip edge, it is total wins/losses:
        self.wins = 0
        self.losses = 0
        # for the number of rounds, we need to remember how many rounds we've played:
        self.roundsElapsed = 0

        # the rest proceeds as before:
        self.action_space = spaces.Discrete(int(maxWealth*100))
        self.observation_space = spaces.Tuple((
            spaces.Box(0, maxWealth, shape=[1]), # current wealth
            spaces.Discrete(maxRounds+1), # rounds elapsed
            spaces.Discrete(maxRounds+1), # wins
            spaces.Discrete(maxRounds+1), # losses
            spaces.Box(0, maxWealth, [1]))) # maximum observed wealth
        self.reward_range = (0, maxWealth)
        self.edge = edge
        self.wealth = self.initialWealth
        self.maxRounds = maxRounds
        self.rounds = self.maxRounds
        self.maxWealth = maxWealth
        if reseed or not hasattr(self, 'np_random') : self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = action/100.0
        if action > self.wealth:
          action = self.wealth
        if self.wealth < 0.000001:
            done = True
            reward = 0.0
        else:
          if self.rounds == 0:
            done = True
            reward = self.wealth
          else:
            self.rounds = self.rounds - 1
            done = False
            reward = 0.0
            coinflip = flip(self.edge, self.np_random)
            self.roundsElapsed = self.roundsElapsed+1
            if coinflip:
              self.wealth = min(self.maxWealth, self.wealth + action)
              self.maxEverWealth = max(self.wealth, self.maxEverWealth)
              self.wins = self.wins+1
            else:
              self.wealth = self.wealth - action
              self.losses = self.losses+1
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (np.array([float(self.wealth)]), self.roundsElapsed, self.wins, self.losses, np.array([float(self.maxEverWealth)]))
    def reset(self):
        # re-init everything to draw new parameters etc, but preserve the RNG for reproducibility and pass in the same hyperparameters as originally specified:
        self.__init__(initialWealth=self.initialWealth, edgePriorAlpha=self.edgePriorAlpha, edgePriorBeta=self.edgePriorBeta, maxWealthAlpha=self.maxWealthAlpha, maxWealthM=self.maxWealthM, maxRoundsMean=self.maxRoundsMean, maxRoundsSD=self.maxRoundsSD, reseed=False)
        return self._get_obs()
    def render(self, mode='human'):
        print("Current wealth: ", self.wealth, "; Rounds left: ", self.rounds, "; True edge: ", self.edge,
              "; True max wealth: ", self.maxWealth, "; True stopping time: ", self.maxRounds, "; Rounds left: ",
              self.maxRounds - self.roundsElapsed)
