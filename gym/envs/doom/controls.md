###Controls

Doom is usually played with a full keyboard, and multiple keys can be pressed at once.

To replicate this, we broke down the possible actions in 40 keys. Each key can be pressed (value of 1), or unpressed (value of 0).

The deltas (35 to 39) indicate speed of change (values 0 to 10), where higher values will make the player move faster on an axis.

A list of values is expected to be passed as the action (e.g. [0, 1, 0, 0, 1, 0, .... ]).

Each map is restricted on what actions can be performed, but the mapping is the same across all maps.

For example, if we want to [0] - ATTACK, [2] - JUMP, and [12] - MOVE_FORWARD at the same time, we would submit the following action:

```python
action = [0] * 40
action[0] = 1
action[2] = 1
action[12] = 1
```

The full list of possible actions is:

* [0]  - ATTACK                           - Shoot weapon - Values 0 or 1
* [1]  - USE                              - Use item - Values 0 or 1
* [2]  - JUMP                             - Jump - Values 0 or 1
* [3]  - CROUCH                           - Crouch - Values 0 or 1
* [4]  - TURN180                          - Perform 180 turn - Values 0 or 1
* [5]  - RELOAD                           - Reload weapon - Values 0 or 1
* [6]  - ZOOM                             - Toggle zoom in/out - Values 0 or 1
* [7]  - SPEED                            - Run faster - Values 0 or 1
* [8]  - STRAFE                           - Strafe (moving sideways in a circle) - Values 0 or 1
* [9]  - MOVE_RIGHT                       - Move to the right - Values 0 or 1
* [10] - MOVE_LEFT                        - Move to the left - Values 0 or 1
* [11] - MOVE_BACKWARD                    - Move backward - Values 0 or 1
* [12] - MOVE_FORWARD                     - Move forward - Values 0 or 1
* [13] - TURN_RIGHT                       - Turn right - Values 0 or 1
* [14] - TURN_LEFT                        - Turn left - Values 0 or 1
* [15] - LOOK_UP                          - Look up - Values 0 or 1
* [16] - LOOK_DOWN                        - Look down - Values 0 or 1
* [17] - LAND                             - Land (e.g. drop from ladder) - Values 0 or 1
* [18] - SELECT_WEAPON1                   - Select weapon 1 - Values 0 or 1
* [19] - SELECT_WEAPON2                   - Select weapon 2 - Values 0 or 1
* [20] - SELECT_WEAPON3                   - Select weapon 3 - Values 0 or 1
* [21] - SELECT_WEAPON4                   - Select weapon 4 - Values 0 or 1
* [22] - SELECT_WEAPON5                   - Select weapon 5 - Values 0 or 1
* [23] - SELECT_WEAPON6                   - Select weapon 6 - Values 0 or 1
* [24] - SELECT_WEAPON7                   - Select weapon 7 - Values 0 or 1
* [25] - SELECT_WEAPON8                   - Select weapon 8 - Values 0 or 1
* [26] - SELECT_WEAPON9                   - Select weapon 9 - Values 0 or 1
* [27] - SELECT_WEAPON0                   - Select weapon 0 - Values 0 or 1
* [28] - SELECT_NEXT_WEAPON               - Select next weapon - Values 0 or 1
* [29] - SELECT_PREV_WEAPON               - Select previous weapon - Values 0 or 1
* [30] - DROP_SELECTED_WEAPON             - Drop selected weapon - Values 0 or 1
* [31] - ACTIVATE_SELECTED_WEAPON         - Activate selected weapon - Values 0 or 1
* [32] - SELECT_NEXT_ITEM                 - Select next item - Values 0 or 1
* [33] - SELECT_PREV_ITEM                 - Select previous item - Values 0 or 1
* [34] - DROP_SELECTED_ITEM               - Drop selected item - Values 0 or 1
* [35] - LOOK_UP_DOWN_DELTA               - Look Up - Values 0 to 10 (Higher value increases speed)
* [36] - TURN_LEFT_RIGHT_DELTA            - Turn left/right - Values 0 to 10 (Higher value increases speed)
* [37] - MOVE_FORWARD_BACKWARD_DELTA      - Move forward/backward - Values 0 to 10 (Higher value increases speed)
* [38] - MOVE_LEFT_RIGHT_DELTA            - Move left/right - Values 0 to 10 (Higher value increases speed)
* [39] - MOVE_UP_DOWN_DELTA               - Move up/down - Values 0 to 10 (Higher value increases speed)
