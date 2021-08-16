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
	env.render()  # by default `mode="human"`(GUI), you can pass `mode="rbg_array"` to retrieve an image instead
	env.step(env.action_space.sample())  # take a random action 	
env.close()
```

The output should look something like this

![cartpole-no-reset](https://user-images.githubusercontent.com/28860173/129241283-70069f7c-453d-4670-a226-854203bd8a1b.gif)


The commonly used methods are: 

`reset()` resets the environment to its initial state and returns the observation corresponding to the initial state
`step(action)` takes an action as an input and implements that action in the environment. This method returns a set of four values 
`render()` renders the environment
	
- `observation` (**object**) : an environment specific object representation your observation of the environment after the step is taken. Its often aliased as the next state after the action has been taken
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
- `observation_space`: this attribute gives the format of valid observations. It if of datatype `Space` provided by Gym. (For ex: if the observation space is of type `Box` and the shape of the object is `(4,)`, this denotes a valid observation will be an array of 4 numbers). We can check the box bounds as well with attributes

```python
print(env.observation_space.high)
#> array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=float32)

print(env.observation_space.low)
#> array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38], dtype=float32)
```
- There are multiple types of Space types inherently available in gym:
	- `Box` describes an n-dimensional continuous space. Its a bounded space where we can define the upper and lower limit which describe the valid values our observations can take.
	- `Discrete` describes a discrete space where { 0, 1, ......., n-1} are the possible values our observation/action can take. 
	- `Dict` represents a dictionary of simple spaces.
	- `Tuple` represents a tuple of simple spaces
	- `MultiBinary` creates a n-shape binary space. Argument n can be a number or a `list` of numbers
	- `MultiDiscrete` consists of a series of `Discrete` action spaces with different number of actions in each element
	```python
	observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
	print(observation_space.sample())
	#> [ 1.6952509 -0.4399011 -0.7981693]

	observation_space = Discrete(4)
	print(observation_space.sample())
	#> 1

	observation_space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
	print(observation_space.sample())
	#> OrderedDict([('position', 0), ('velocity', 1)])

	observation_space = Tuple((Discrete(2), Discrete(3)))
	print(observation_space.sample())
	#> (1, 2)

	observation_space = MultiBinary(5)
	print(observation_space.sample())
	#> [1 1 1 0 1]

	observation_space = MultiDiscrete([ 5, 2, 2 ])
	print(observation_space.sample())
	#> [3 0 0]
	```
- `reward_range`:  returns a tuple corresponding to min and max possible rewards. Default range is set to `[-inf,+inf]`. You can set it if you want a narrower range 
- `close()` : Override close in your subclass to perform any necessary cleanup
- `seed()`: Sets the seed for this env's random number generator


### Unwrapping an environment
If you have a wrapped environment, and you want to get the unwrapped environment underneath all the layers of wrappers (so that you can manually call a function or change some underlying aspect of the environment), you can use the `.unwrapped` attribute. If the environment is already a base environment, the `.unwrapped` attribute will just return itself.

```python
base_env = env.unwrapped
```

### Vectorized Environment
Vectorized Environments are a way of stacking multiple independent environments, so that instead of training on one environment, our agent can train on multiple environments at a time. Each `observation` returned from a vectorized environment is a batch of observations for each sub-environment, and `step` is also expected to receive a batch of actions for each sub-environment.

**NOTE:** All sub-environments should share the identical observation and action spaces. A vector of multiple different environments is not supported

Gym Vector API consists of two types of vectorized environments:

- `AsyncVectorEnv` runs multiple environments in parallel. It uses `multiprocessing` processes, and pipes for communication.
- `SyncVectorEnv`runs multiple environments serially

```python
import gym
env = gym.vector.make('CartPole-v1', 3,asynchronous=True)  # Creates an Asynchronous env
env.reset()
#> array([[-0.04456399, 0.04653909, 0.01326909, -0.02099827],
#> [ 0.03073904, 0.00145001, -0.03088818, -0.03131252],
#> [ 0.03468829, 0.01500225, 0.01230312, 0.01825218]],
#> dtype=float32)

```
