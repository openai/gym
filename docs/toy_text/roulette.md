---
action-type: "Discrete"
title: "Roulette"
actions: Discrete
agents: "1"
manual-control: "Yes"
action-shape: "(1,)"
action-values: "(0,1)"
observation-shape: "(1,)"
observation-values: "(0,4)"
average-total-reward: ""
import: "from gym.toy_text import roulette"
agent-labels: "none"
---


The roulette wheel has <a href="#spots">spots</a> spots. If the bet is 0 and a 0 comes up,
you win a reward of <a href="#spots">spots</a>-2. If the parity of your bet matches the parity
of the spin, you win 1. Otherwise you receive a reward of -1.
The long run reward for playing 0 should be -1/<a href="#spots">spots</a> for any state
The last action (<a href="#spots">spots</a>+1) stops the rollout for a return of 0 (walking away)

**Rewards:**

Reward schedule:
- <a href="#spots">spots</a>-2: reward for betting and landing a 0.
- 1: reward for having same parity as landed number.
- -1/<a href="#spots">spots</a>: long run reward for playing 0.


### Manual Control

TBD


### Arguments

```
gym.make('Roulette-v0', spots=37)
```

<a id="spots">`spots`</a>: Number of spots on Roulette.



### Version History

* v0: Initial versions release (1.0.0)
