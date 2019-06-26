import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)?
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  
    The player is playing against a dealer with a fixed strategy. 
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is played with an infinite deck (or with replacement).
    The game starts with the player and dealer each receiving two cards. 
    One of the dealer's cards is facedown and the other is visible. 

    The player can request additional cards (action=1, hit) until they decide to stop
    (action=0, stick) or exceed 21 (bust). If double down is flagged (double_down=True), 
    the player can double their bet (action=2, double) and then will receive exactly one 
    additional card.

    If the player is dealt a 10 or face card and an Ace, this is called a 
    natural blackjack and if natural is flagged (natural=True), the player immediately
    wins a payout of 1.5 (reward=1.5) unless the dealer also has a natural Blackjack. 

    If the player busts, they immediately lose (reward=-1). After a stick or 
    double down that does not result in a bust, the dealer draws until their sum 
    is 17 or greater. If the dealer busts, the player wins (reward=1). Rewards 
    are doubled in the double down case. 

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1. These are again doubled in the double down
    case. 

    The observation is a 3-tuple of: the player's current sum,
    the dealer's one showing card (1-10 where 1 is Ace),
    and whether or not the player holds a usable Ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    def __init__(self, natural=False, double_down=False):
        if double_down:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.seed()
        self.natural = natural
        self.double_down = double_down
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.natural and is_natural(self.player):
            if is_natural(self.dealer):
                reward = 0.0 #player and dealer natural Blackjack
            else:
                reward = 1.5 #player natural Blackjack
            done = True
            return self._get_obs(), reward, done, {}

        assert self.action_space.contains(action)
        if action == 2: # double down: bet double and get only 1 card, then compare
            self.player.append(draw_card(self.np_random))
            done = True
            if is_bust(self.player):
                reward = -2.0
            else: 
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card(self.np_random))
                reward = 2 * cmp(score(self.player), score(self.dealer))
        elif action == 1:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        elif action == 0:  # stick: play out the dealer's hand, then compare
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()
