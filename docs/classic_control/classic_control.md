---
layout: env_selection
title: Classic Control Environments
---
<div class="selection-content" markdown="1">

The unique dependencies for this set of environments can be installed via:

````bash
pip install gym[classic_control]
````

There are five classic control environments: Acrobot, CartPole, Mountain Car, Continuous Mountain Car, and Pendulum. All of these environments are stochastic in terms of their initial state, within a given range. In addition, Acrobot has noise applied to the taken action. Also, regarding the both mountain car environments, the cars are under powered to climb the mountain, so it takes some effort to reach the top.

Among Gym environments, this set of environments can be considered as easier ones to solve by a policy.

All environments are highly configurable via arguments specified in each environment's documentation.

</div>
