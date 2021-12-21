Mountain Car
---
|Title|Action Type|Action Shape|Action Values|Observation Shape|Observation Values|Average Total Reward|Import|
| ----------- | -----------| ----------- | -----------| ----------- | -----------| ----------- | -----------|
|Mountain Car|Discrete|(1,)|(0,1,2)|(2,)|[(-1.2,0.6),(-0.07,0.07)]| |`from gym.envs.classic_control.mountain_car import MountainCarEnv`|
---

The agent (a car) is started at the bottom of a valley. For any given state the agent may choose to accelerate to the left, right or cease any acceleration. The code is originally based on [this code](http://incompleteideas.net/MountainCar/MountainCar1.cp) and the environment appeared first in Andrew Moore's PhD Thesis (1990):
```
@TECHREPORT{Moore90efficientmemory-based,
    author = {Andrew William Moore},
    title = {Efficient Memory-based Learning for Robot Control},
    institution = {},
    year = {1990}
}
```

Observation space is a 2-dim vector, where the 1st element represents the "car position" and the 2nd element represents the "car velocity".

There are 3 discrete deterministic actions:
- 0: Accelerate to the Left
- 1: Don't accelerate
- 2: Accelerate to the Right

Reward: Reward of 0 is awarded if the agent reached the flag (position = 0.5) on top of the mountain. Reward of -1 is awarded if the position of the agent is less than 0.5.

Starting State: The position of the car is assigned a uniform random value in [-0.6 , -0.4]. The starting velocity of the car is always assigned to 0.

Episode Termination: The car position is more than 0.5. Episode length is greater than 200
         


### Arguments

```
gym.make('MountainCar-v0')
```

### Version History

* v0: Initial versions release (1.0.0)
