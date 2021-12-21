Continuous Mountain Car
---
|Title|Action Type|Action Shape|Action Values|Observation Shape|Observation Values|Average Total Reward|Import|
| ----------- | -----------| ----------- | -----------| ----------- | -----------| ----------- | -----------|
|Continuous Mountain Car|Continuous|(1,)|[(-1.0,1.0)]|(2,)|[(-1.2,0.6),(-0.07,0.07)]| |`from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv`|
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

Action: The actual driving force is calculated by multiplying the power coef by power (0.0015)

Reward: Reward of 100 is awarded if the agent reached the flag (position = 0.45) on top of the mountain. Reward is decrease based on amount of energy consumed each step.

Starting State: The position of the car is assigned a uniform random value in [-0.6 , -0.4]. The starting velocity of the car is always assigned to 0.

Episode Termination: The car position is more than 0.45. Episode length is greater than 200
         


### Arguments

```
gym.make('MountainCarContinuous-v0')
```

### Version History

* v0: Initial versions release (1.0.0)
