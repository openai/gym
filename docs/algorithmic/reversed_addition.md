Reversed Addition
---
|Title|Action Type|Action Shape|Action Values|Observation Shape|Observation Values|Average Total Reward|Import|
| ----------- | -----------| ----------- | -----------| ----------- | -----------| ----------- | -----------|
|Reversed Addition|Discrete|(3,)|[(0,1,2,3),(0,1),(0,<a href="#base">base</a>-1)]|(1,)|(0,<a href="#base">base</a>)| |from gym.envs.algorithmic import reversed_addition|
---

The goal is to add <a href="#rows">"rows"</a> number of multi-digit sequences, provided on an input grid. The sequences are provided in <a href="#rows">"rows"</a> number adjacent rows, with the right edges aligned. The initial position of the read head is the last digit of the top number (i.e. upper-right corner). This task was originally used in the paper <a href="http://arxiv.org/abs/1511.07275">Learning Simple Algorithms from Examples</a>.

The model has to: 
- memorize an addition table for pairs of digits. 
- learn how to move over the input grid.
- discover the concept of a carry. 

The agent take a 3-element vector for actions.
The action space is `(x, w, v)`, where: 
- `x` is used for direction of movement. It can take values (0,1,2,3).
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
gym.make('ReversedAddition-v0', rows=2, base=3) #for ReversedAddition
gym.make('ReversedAddition3-v0', rows=3, base=3) #for ReversedAddition3
gym.make('ReversedAddition-v0', rows=n, base=3) #for ReversedAddition with n numbers
```

<a id="rows">`rows`</a>: Number of multi-digit sequences to add at a time.

<a id="base">`base`</a>: Number of distinct characters to read/write.

### Version History

* v0: Initial versions release (1.0.0)
