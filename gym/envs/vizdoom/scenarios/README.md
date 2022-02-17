# Scenarios  Decription:

Scenarios contained in iwad files do not support action constraints, death penalty and living rewards.
Every mention of any settings that are not included in iwads will be specified with "(config)".

Note: Vizdoom does not support setting certain rewards (such as killing oponents) in .cfg files. These must be set in the .wad files instead

## BASIC
The purpose of the scenario is just to check if using this
framework to train some AI in 3D environment is feasible.

Map is a rectangle with gray walls, ceiling and floor.
Player is spawned along the longer wall, in the center.
A red, circular monster is spawned randomly somewhere along
the opposite wall. Player can only (config) go left/right 
and shoot. 1 hit is enough to kill the monster. Episode 
finishes when monster is killed or on timeout.

__REWARDS:__

+101 for killing the monster
-5 for missing
Episode ends after killing the monster or on timeout.

Further configuration:
* living reward = -1,
* 3 available buttons: move left, move right, shoot (attack)
* timeout = 300

## DEADLY CORRIDOR
The purpose of this scenario is to teach the agent to navigate towards
his fundamental goal (the vest) and make sure he survives at the 
same time.

Map is a corridor with shooting monsters on both sides (6 monsters 
in total). A green vest is placed at the oposite end of the corridor.
Reward is proportional (negative or positive) to change of the
distance between the player and the vest. If player ignores monsters 
on the sides and runs straight for the vest he will be killed somewhere 
along the way. To ensure this behavior doom_skill = 5 (config) is 
needed.

__REWARDS:__

+dX for getting closer to the vest.
-dX for getting further from the vest.

Further configuration:
* 7 available buttons: turn left, turn right, move forward, move backward, move left, move right, shoot (attack)
* timeout = 2100
* death penalty = 100
* doom_skill = 5


## DEFEND THE CENTER
The purpose of this scenario is to teach the agent that killing the 
monsters is GOOD and when monsters kill you is BAD. In addition,
wasting amunition is not very good either. Agent is rewarded only 
for killing monsters so he has to figure out the rest for himself.

Map is a large circle. Player is spawned in the exact center.
5 melee-only, monsters are spawned along the wall. Monsters are 
killed after a single shot. After dying each monster is respawned 
after some time. Episode ends when the player dies (it's inevitable 
becuse of limitted ammo).

__REWARDS:__
+1 for killing a monster

Further configuration:
* 3 available buttons: turn left, turn right, shoot (attack)
* death penalty = 1

## DEFEND THE LINE
The purpose of this scenario is to teach the agent that killing the 
monsters is GOOD and when monsters kill you is BAD. In addition,
wasting amunition is not very good either. Agent is rewarded only 
for killing monsters so he has to figure out the rest for himself.

Map is a rectangle. Player is spawned along the longer wall, in the 
center. 3 melee-only and 3 shooting monsters are spawned along the 
oposite wall. Monsters are killed after a single shot, at first. 
After dying each monster is respawned after some time and can endure 
more damage. Episode ends when the player dies (it's inevitable 
becuse of limitted ammo).

__REWARDS:__
+1 for killing a monster

Further configuration:
* 3 available buttons: turn left, turn right, shoot (attack)
* death penalty = 1

## HEALTH GATHERING
The purpose of this scenario is to teach the agent how to survive
without knowing what makes him survive. Agent know only that life 
is precious and death is bad so he must learn what prolongs his 
existence and that his health is connected with it.

Map is a rectangle with green, acidic floor which hurts the player
periodically. Initially there are some medkits spread uniformly
over the map. A new medkit falls from the skies every now and then.
Medkits heal some portions of player's health - to survive agent 
needs to pick them up. Episode finishes after player's death or 
on timeout.


Further configuration:
* living_reward = 1
* 3 available buttons: turn left, turn right, move forward
* 1  available game variable: HEALTH
* death penalty = 100

## MY WAY HOME
The purpose of this scenario is to teach the agent how to navigate
in a labirynth-like surroundings and reach his ultimate goal 
(and learn what it actually is).

Map is a series of rooms with interconnection and 1 corridor 
with a dead end. Each room has a different color. There is a 
green vest in one of the rooms (the same room every time). 
Player is spawned in randomly choosen room facing a random 
direction. Episode ends when vest is reached or on timeout/

__REWARDS:__
+1 for reaching the vest

Further configuration:
* 3 available buttons: turn left, turn right, move forward
* living reward = -0.0001
* timeout = 2100

## PREDICT POSITION
The purpose of the scenario is teach agent to synchronize 
missle weapon shot (involving a signifficant delay between 
shooting and hitting) with target movements. Agent should be 
able to shoot so that missle and monster meet each other.

The map is a rectangle room. Player is spawned along the longer 
wall, in the center. A monster is spawned randomly somewhere 
along the opposite wall and walks between left and right corners 
along the wall. Player is equipped with a rocket launcher and 
a single rocket. Episode ends when missle hits a wall/the monster 
or on timeout.

__REWARDS:__
+1 for killing the monster

Further configuration:
* living reward = -0.0001,
* 3 available buttons: move left, move right, shoot (attack)
* timeout = 300

## TAKE COVER
The purpose of this scenario is to teach agent to link incomming 
missles with his estimated lifespan. Agent should learn that 
being hit means health decrease and this in turn will lead to
death which is undesirable. In effect agent should avoid 
missles.

Map is a rectangle. Player is spawned along the longer wall, 
in the center. A couple of shooting monsters are spawned 
randomly somewhere along the opposite wall and try to kill 
the player with fireballs. The player can only (config) move 
left/right. More monsters appear with time. Episode ends when 
player dies.

__REWARDS:__
+1 for each tic of life

Further configuration:
* living reward = 1.0,
* 2 available buttons: move left, move right
