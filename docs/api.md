# API
## Initializing Environments
Initializing environment is very easy in Gym and can be done via: 

```python
import gym
env = gym.make('CartPole-v0')
```

## Interacting with the Environment
This example will run an instance of `CartPole-v0` environment for 1000 timesteps, rendering the environment at each step. You should see a window pop up rendering the classic [cart-pole](https://www.youtube.com/watch?v=J7E6_my3CHk&ab_channel=TylerStreeter) problem

```python
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000): 
	env.render() 
	env.step(env.action_space.sample()) # take a random action 	
env.close()
```

The commonly used methods are: 

`reset()` resets the environment to its initial state and returns the observation corresponding to the initial state
`step(action)` takes an action as an input and implements that action in the environement. This method returns a set of four values 
`render()` renders the environment
	
- `observation` (**object**) : an environment specific object repesentation your observation of the environment after the step is taken. Its often aliased as the next state after the action has been taken
- `reward`(**float**) : immediate reward achieved by the previous action. Actual value and range will varies between environments, but the final goal is always to increase your total reward
- `done`(**boolean**): whether it’s time to `reset` the environment again. Most (but not all) tasks are divided up into well-defined episodes, and `done` being `True` indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
- `info`(**dict**) : This provides general information helpful for debugging or additional information depending on the environment, such as the raw probabilities behind the environment’s last state change


## Additional Environment API

- `action_space`: this attribute gives the format of valid actions. It is of datatype `Space` provided by Gym. (For ex: If the action space is of type `Discrete` and gives the value `Discrete(2)`, this means there are two valid discrete actions 0 & 1 )
```python
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)

```
- `observation_space`this attribute gives the format of valid observations. It if of datatype `Space` provided by Gym. (For ex: if the observation space is of type `Box` and the shape of the object is `(4,)`, this denotes a valid observation will be an array of 4 numbers). We can check the box bounds as well with attributes

```python
print(env.observation_space.high)
#> array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=float32)
print(env.observation_space.low)
#> array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38], dtype=float32)
```
- `reward_range` returns a tuple corresponding to min and max possible rewards. Default range is set to `[-inf,+inf]`. You can set it if you want a narrower range 
- `close()` : Override close in your subclass to perform any necessary cleanup
- `seed()`: Sets the seed for this env's random number generator


### Unwrapping an environment
If you have a wrapped environment, and you want to get the unwrapped environment underneath all the layers of wrappers (so that you can manually call a function or change some underlying aspect of the environment), you can use the `.unwrapped` attribute. If the environment is already a base environment, the `.unwrapped` attribute will just return itself.

```python
base_env = env.unwrapped
```

