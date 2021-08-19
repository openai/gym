# Algorithmic Environments

The unique dependencies for this set of environments can be installed via:

````bash
pip install gym[algorithmic]
````

### Characteristics

Algorithmic environments have the following traits in common:
- A 1-d "input tape" or 2-d "input grid" of characters
- A target string which is a deterministic function of the input characters

Agents control a read head that moves over the input tape. Observations consist
of the single character currently under the read head. The read head may fall
off the end of the tape in any direction. When this happens, agents will observe
a special blank character (with index=env.base) until they get back in bounds.

### Actions
Actions consist of 3 sub-actions:
- Direction to move the read head (left or right, plus up and down for 2-d
      envs)
- Whether to write to the output tape
- Which character to write (ignored if the above sub-action is 0)

An episode ends when:
- The agent writes the full target string to the output tape.
- The agent writes an incorrect character.
- The agent runs out the time limit. (Which is fairly conservative.)

Reward schedule:
- write a correct character: +1
- write a wrong character: -.5
- run out the clock: -1
- otherwise: 0

In the beginning, input strings will be fairly short. After an environment has
been consistently solved over some window of episodes, the environment will
increase the average length of generated strings. Typical env specs require
leveling up many times to reach their reward threshold.
