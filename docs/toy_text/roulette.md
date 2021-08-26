Roulette
---
|Title|Action Type|Action Shape|Action Values|Observation Shape|Observation Values|Average Total Reward|Import|
| ----------- | -----------| ----------- | -----------| ----------- | -----------| ----------- | -----------|
|Roulette|Discrete|(1,)|(0,1)|(1,)|(0,4)| |from gym.envs.toy_text import roulette|
---


The roulette wheel has <a href="#spots">spots</a> spots. If the bet is 0 and a 0 comes up, you win a reward of <a href="#spots">spots</a>-2. If any other number comes up you get a reward of -1.

For non-zero bets, if the parity of your bet matches the parity of the spin, you win 1. Otherwise you receive a reward of -1.

The last action (<a href="#spots">spots</a>+1) stops the rollout for a return of 0 (walking away)

**Rewards:**

Reward schedule:
- <a href="#spots">spots</a>-2: reward for betting and landing a 0.
- 1: For non-zero bets, reward for having same parity as landed number.
- -1: For non-zero bets, reward for having different parity as landed number and for betting zero and landing a non-zero number.

### Arguments

```
gym.make('Roulette-v0', spots=37)
```

<a id="spots">`spots`</a>: Number of spots on Roulette.



### Version History

* v0: Initial versions release (1.0.0)
