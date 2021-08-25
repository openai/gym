Frozen Lake---
|Title|Action Type|Action Shape|Action Values|Observation Shape|Observation Values|Average Total Reward|Import|
| ----------- | -----------| ----------- | -----------| ----------- | -----------| ----------- | -----------|
|Frozen Lake|Continuous|(1,)|(-10000, 10000)|(1,)|(0,3)| |from gym.envs.toy_text import guessing_game|
---

The object of the game is to guess within 1% of the randomly chosen number
within 200 time steps.

The agent take a 1-element vector for actions.
The action space is `(action)`, where: 
- `action` is the prediction of the agent.

After each step the agent is provided with one of four possible observations
which indicate where the guess is in relation to the randomly chosen number:

- 0 - No guess yet submitted (only after reset)
- 1 - Guess is lower than the target
- 2 - Guess is equal to the target
- 3 - Guess is higher than the target

The episode terminates after the agent guesses within 1% of the target or
200 steps have been taken.

The agent will need to use a memory of previously submitted actions and observations in order to efficiently explore the available actions.
The purpose is to have agents optimize their exploration parameters (e.g. how far to explore from previous actions) based on previous experience. Because the goal changes each episode a state-value or action-value function isn't able to provide any additional benefit apart from being able to tell whether to increase or decrease the next guess.

The perfect agent would likely learn the bounds of the action space (without referring
to them explicitly) and then follow binary tree style exploration towards to goal number

**Rewards:**

Reward schedule:
- 0 if the agent's guess is outside of 1% of the target
- 1 if the agent's guess is inside 1% of the target

### Arguments

```
gym.make('GuessingGame-v0' )
```

### Version History

* v0: Initial versions release (1.0.0)
