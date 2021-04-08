import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete
import numpy as np


class NChainEnv(discrete.DiscreteEnv):
    """n-Chain environment

    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward

    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.

    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.

    The observed state is the current state in the chain (0 to n-1).

    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf

    >>> env = NChainEnv()
    >>> print(env.action_space, env.observation_space)
    Discrete(2) Discrete(5)

    >>> env.P
    {0: {0: [(0.8, 1, 0, False), (0.2, 0, 2, False)], 1: [(0.8, 0, 2, False), (0.2, 1, 0, False)]}, 1: {0: [(0.8, 2, 0, False), (0.2, 0, 2, False)], 1: [(0.8, 0, 2, False), (0.2, 2, 0, False)]}, 2: {0: [(0.8, 3, 0, False), (0.2, 0, 2, False)], 1: [(0.8, 0, 2, False), (0.2, 3, 0, False)]}, 3: {0: [(0.8, 4, 0, False), (0.2, 0, 2, False)], 1: [(0.8, 0, 2, False), (0.2, 4, 0, False)]}, 4: {0: [(0.8, 4, 10, False), (0.2, 0, 2, False)], 1: [(0.8, 0, 2, False), (0.2, 4, 10, False)]}}

    This environment starting state is always 0
    >>> env.s
    0

    >>> _ = env.seed(42) # 12

    Take a step forward (0), we get 0 reward, and ends up in state 1
    >>> env.step(0)
    (1, 0, False, {'prob': 0.8})

    We are really in state 1
    >>> env.s
    1

    And the previous action was 0 (forward)
    >>> env.lastaction
    0

    We can quickly reset, so that current state is 0 and last action is none
    >>> env.reset()
    0
    >>> env.s
    0
    >>> env.lastaction

    Take a step backward, we we slip forward
    >>> env.step(1)
    (1, 0, False, {'prob': 0.2})

    Take a step backward, we get small reward (2), and ends up in state 0
    >>> env.step(1)
    (0, 2, False, {'prob': 0.8})

    We are really in state 0
    >>> env.s
    0

    Always trying to step forward
    >>> step_forward_10 = [env.step(0) for i in range(10)]
    
    We may slip back once in a while and ends up in strating state (0)
    >>> [s[0] for s in step_forward_10]
    [1, 2, 3, 4, 4, 4, 4, 0, 1, 0]

    The forward reward is 0 unless we are already at last state, or 2 if we (intentionally or accidentally) go backward
    >>> [s[1] for s in step_forward_10]
    [0, 0, 0, 0, 10, 10, 10, 2, 0, 2]

    No state, even last one, is exit/done state
    >>> [s[2] for s in step_forward_10]
    [False, False, False, False, False, False, False, False, False, False]
    """
    def __init__(self, n=5, slip=0.2, small=2, large=10):
        start_state = 0
        forward_reward = 0
        action_forward = 0
        action_backward = 1
        last_state = n - 1

        nS = n
        nA = 2
        # We always start in state 0
        isd = np.zeros(nS)
        isd[start_state] = 1.0

        P = {s: {
                action_forward: [
                    (1-slip, s+1, forward_reward, False),
                    (slip, start_state, small, False),
                ],
                action_backward: [
                    (1-slip, start_state, small, False),
                    (slip, s+1, forward_reward, False),
                ],
            } for s in range(nS - 1)}

        P[last_state] = {
                action_forward: [
                    (1-slip, last_state, large, False),
                    (slip, start_state, small, False),
                ],
                action_backward: [
                    (1-slip, start_state, small, False),
                    (slip, last_state, large, False),
                ],
            }
        
        super(NChainEnv, self).__init__(nS, nA, P, isd)
