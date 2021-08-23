NChain
---
|Title|Action Type|Action Shape|Action Values|Observation Shape|Observation Values|Average Total Reward|Import|
| ----------- | -----------| ----------- | -----------| ----------- | -----------| ----------- | -----------|
|NChain|Discrete|(1,)|(0,1)|(1,)|(0,4)| |from gym.envs.toy_text import nchain|
---

This game presents moves along a linear chain of states, with two actions:

- 0- forward, which moves along the chain but returns no reward
- 1- backward, which returns to the beginning and has a small reward

The end of the chain, however, presents a large reward, and by moving 'forward' at the end of the chain this large reward can be repeated.

At each action, there is a small probability that the agent 'slips' and the opposite transition is instead taken.

The observed state is the current state in the chain (0 to <a href="#n">n</a>-1).

This environment is described in section 6.1 of:
A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf

**Rewards:**

Reward schedule:
- <a href="#small">small</a> - reward for performing backward action.
- <a href="#large">large</a> - reward for reaching end of chain.

### Arguments

```
gym.make('NChain-v0', n=5, slip=0.2, small=2, large=10)
```

<a id="n">`n`</a>: Length of chain.

`slip`: Probability of performing reverse action.

<a id="small">`small`</a>: Reward for moving backwards.

<a id="large">`large`</a>: Reward on reaching end of chain.



### Version History

* v0: Initial versions release (1.0.0)
