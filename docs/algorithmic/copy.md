---
action-type: "Discrete"
title: "Copy"
actions: Discrete
agents: "1"
manual-control: "Yes"
action-shape: "(3,)"
action-values: "[(0, 1),(0,1),(0,<a href="#base">base</a>-1)]"
observation-shape: "(1,)"
observation-values: "(0,<a href="#base">base</a>)"
average-total-reward: ""
import: "from gym.algorithmic import Copy-v0"
agent-labels: "none"
---

{% include info_box.md %}

This task involves copying content from the input tape to the output tape. This task was originally used in the paper <a href="http://arxiv.org/abs/1511.07275">Learning Simple Algorithms from Examples</a>.

The model has to learn: 
- correspondence between input and output symbols.
- executing the move right action on input tape.

The agent take a 3-element vector for actions.
The action space is `(x, w, v)`, where: 
- `x` is used for left/right movement. It can take values (0,1).
- `w` is used for writing to output tape or not. It can take values (0,1). 
- `r` is used for selecting the value to be written on output tape.


The observation space size is `(1,)` .

**Rewards:**

Rewards are issued similar to other Algorithmic Environments.Reward schedule:
- write a correct character: +1
- write a wrong character: -.5
- run out the clock: -1
- otherwise: 0

### Manual Control

TBD


### Arguments

```
gym.make('Copy-v0', base=5, chars=True)
```

<a id="base">`base`</a>: Number of distinct characters to read/write.

`chars`: If True, use uppercase alphabet. Otherwise, digits. Only affects rendering.

### Version History

* v0: Initial versions release (1.0.0)
