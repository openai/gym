Reverse
---
|Title|Action Type|Action Shape|Action Values|Observation Shape|Observation Values|Average Total Reward|Import|
| ----------- | -----------| ----------- | -----------| ----------- | -----------| ----------- | -----------|
|Reverse|Discrete|(3,)|[(0, 1),(0,1),(0,<a href="#base">base</a>-1)]|(1,)|(0,<a href="#base">base</a>)| |from gym.envs.algorithmic import reverse|
---

The goal is to reverse a sequence of symbols on the input tape. We provide a special character `r` to indicate the end of the sequence. The model must learn to move right multiple times until it hits the `r` symbol, then move to the left, copying the symbols to the output tape. This task was originally used in the paper <a href="http://arxiv.org/abs/1511.07275">Learning Simple Algorithms from Examples</a>.

The model has to learn: 
- correspondence between input and output symbols.
- executing the move left and right action on input tape.

The agent take a 3-element vector for actions.
The action space is `(x, w, v)`, where: 
- `x` is used for left/right movement. It can take values (0,1).
- `w` is used for writing to output tape or not. It can take values (0,1). 
- `r` is used for selecting the value to be written on output tape.


The observation space size is `(1,)` .

**Rewards:**

Rewards are issued similar to other Algorithmic Environments. Reward schedule:
- write a correct character: +1
- write a wrong character: -.5
- run out the clock: -1
- otherwise: 0

### Arguments

```
gym.make('Reverse-v0', base=2)
```

<a id="base">`base`</a>: Number of distinct characters to read/write.

### Version History

* v0: Initial versions release (1.0.0)
