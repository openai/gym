# POMDP Envs

A generic POMDP implementation.

It supports an underlying MDP with additional clutter (random) state variables.
The observables are then produced by multiplying the cluttered state vector with a confusion matrix:

    Obs = (I - Rand(square: size = #Clutters + #States)) * (Clutter_vec .concat. State_vec)

The environment also supports two separate sets of `good` and `bad` terminal states. Entering such states
will cause proper rewards. The reward signal is therefore computed using the following scheme:

    +1.0  if entering a good terminal state
    -1.0  if entering a bad terminal state
    -1.0  if reaching max_move before reaching a terminal state
    -1.0/max_move  otherwise

Additionally, GenericPOMDPEnv supports partial observability. Any state in `unobservable_states` will be removed
from above equation for computing Obs and the size of matrices will be set accordingly.

Note: In order to have a MDP, simply use `confusion_level=0.0`, `clutter_dim=0`, and `unobservable_states=[]`.

## RestaurantSeekingDialog

The underlying MDP for the instantiated env is a simple user-simulator in a "restaurant-seeking" dialogue system. The POMDP then extends it by doubling the state-space with random clutter and adds uncertainty into the cluttered just as the base class.  

There are three constraints that the user has in mind as her goal: 0: food_type, 1: price_range, and 2: area. The agent starts with no information about the user goal and it has four actions: 

```
0: ask about constraint 0 (food_type), 
1: ask about constraint 1 (price_range)
2: ask about constraint 2 (area)
3: offer a restaurant
```

In this simple MDP, it is assumed that the offer can only be acceptable if the agent knows about all the three user constraints (i.e., if the agent has already asked the three of them). Therefore, the following state exists:

```
0: no information about constraints
1: constraint 0 is known
2: constraint 1 is known
3: constraint 2 is known
4: constraint 0 and 1 are known
5: constraint 0 and 2 are known
6: constraint 1 and 2 are known
7: constraint 0 and 1 and 2 are known
8: a non-acceptable offer has been made (bad terminal)
9: an acceptable offer has been made (good terminal)
```

The state transitions are deterministic and can be defined as the following (s, a, s', p):

```
[[0, 0, 1, 1.],
[0, 1, 2, 1.],
[0, 2, 3, 1.],
[1, 1, 4, 1.],
[1, 2, 5, 1.],
[2, 0, 4, 1.],
[2, 2, 6, 1.],
[3, 0, 5, 1.],
[3, 1, 6, 1.],
[4, 2, 7, 1.],
[5, 1, 7, 1.],
[6, 0, 7, 1.],
[0, 3, 8, 1.],
[1, 3, 8, 1.],
[2, 3, 8, 1.],
[3, 3, 8, 1.],
[4, 3, 8, 1.],
[5, 3, 8, 1.],
[6, 3, 8, 1.],
[7, 3, 9, 1.]]
```

To see the corresponding graph of the MDP, simply do the following:

```
env = gym.make('RestaurantSeekingDialog-v0')
env.reset()
env.write_mdp_to_dot(file_path='mdp.dot')
```

In the `dot` file, `blue diamond` is the initial state and `green squares` and `red squares` are the `good` and `bad` terminals, respectively. In this environment there is only one `good` terminal (`9`) and one `bad` terminal (`8`). The caption on the arrows illustrate corresponding action and transition probability. In order to make an image, you can for example use: `dot -T png -O mdp.dot`.
