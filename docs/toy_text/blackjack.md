Blackjack
---
|Title|Action Type|Action Shape|Action Values|Observation Shape|Observation Values|Average Total Reward|Import|
| ----------- | -----------| ----------- | -----------| ----------- | -----------| ----------- | -----------|
|Blackjack|Discrete|(1,)|(0,1)|(3,)|[(0,31),(0,10),(0,1)]| |from gym.envs.toy_text import blackjack|
---

Blackjack is a card game where the goal is to obtain cards that sum to as near as possible to 21 without going over.  They're playing against a fixed dealer.

Card Values:

- Face cards (Jack, Queen, King) have point value 10.
- Aces can either count as 11 or 1, and it's called 'usable ace' at 11.
- Numerical cards (2-9) have value of their number.

This game is placed with an infinite deck (or with replacement).
The game starts with dealer having one face up and one face down card, while player having two face up cards. 

The player can request additional cards (hit, action=1) until they decide to stop
(stick, action=0) or exceed 21 (bust).
After the player sticks, the dealer reveals their facedown card, and draws
until their sum is 17 or greater.  If the dealer goes bust the player wins.
If neither player nor dealer busts, the outcome (win, lose, draw) is
decided by whose sum is closer to 21.

The agent take a 1-element vector for actions.
The action space is `(action)`, where: 
- `action` is used to decide stick/hit for values (0,1).

The observation of a 3-tuple of: the players current sum,
the dealer's one showing card (1-10 where 1 is ace), and whether or not the player holds a usable ace (0 or 1).

This environment corresponds to the version of the blackjack problem
described in Example 5.1 in Reinforcement Learning: An Introduction
by Sutton and Barto.
http://incompleteideas.net/book/the-book-2nd.html

**Rewards:**

Reward schedule:
- win game: +1
- lose game: -1
- draw game: 0
- win game with natural blackjack: 

    +1.5 (if <a href="#nat">natural</a> is True.) 
    
    +1 (if <a href="#nat">natural</a> is False.)

### Arguments

```
gym.make('Blackjack-v0', natural=False)
```

<a id="nat">`natural`</a>: Whether to give an additional reward for starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

### Version History

* v0: Initial versions release (1.0.0)
