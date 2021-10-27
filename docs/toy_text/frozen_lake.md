Frozen Lake
---
|Title|Action Type|Action Shape|Action Values|Observation Shape|Observation Values|Average Total Reward|Import|
| ----------- | -----------| ----------- | -----------| ----------- | -----------| ----------- | -----------|
|Frozen Lake|Discrete|(1,)|(0,3)|(1,)|(0,nrows*ncolumns)| |from gym.envs.toy_text import frozen_lake|
---


Frozen lake involves crossing a frozen lake from Start(S) to goal(G) without falling into any holes(H). The agent may not always move in the intended direction due to the slippery nature of the frozen lake.

The agent take a 1-element vector for actions.
The action space is `(dir)`, where `dir` decides direction to move in which can be:

- 0: LEFT
- 1: DOWN
- 2: RIGHT
- 3: UP 

The observation is a value representing the agents current position as

    current_row * nrows + current_col

**Rewards:**

Reward schedule:
- Reach goal(G): +1
- Reach hole(H): 0

### Arguments

```
gym.make('FrozenLake-v0', desc=None,map_name="4x4", is_slippery=True)
```

`desc`: Used to specify custom map for frozen lake. For example,

    desc=["SFFF", "FHFH", "FFFH", "HFFG"].

`map_name`: ID to use any of the preloaded maps.

    "4x4":[
        "SFFF", 
        "FHFH", 
        "FFFH", 
        "HFFG"
        ]

    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]


    

`is_slippery`: True/False. If True will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.

    For example, if action is left and is_slippery is True, then:
    - P(move left)=1/3
    - P(move up)=1/3
    - P(move down)=1/3 
### Version History

* v0: Initial versions release (1.0.0)
