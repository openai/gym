# Spaces

Spaces define the valid format of observation and action spaces for an environment. The following spaces are provided:

- [Box](#Box)
- [Discrete](#Discrete)
- [MultiBinary](#MultiBinary)
- [MultiDiscrete](#MultiDiscrete)
- [Dict](#Dict)
- [Tuple](#Tuple)

Each space implements the following functions:

### sample()
Randomly sample an element of this space. Can be
uniform or non-uniform sampling based on boundedness of space.

### contains(x)
Return boolean specifying if x is a valid member of this space.

### _property_ shape
Return the shape of the space as an immutable property.

### _property_ dtype
Return the data type of this space.

### seed(seed)
Seed the PRNG of this space.

### _property_ np_random
Returns the random number generator used by this space.

### to_jsonable(sample_n)
Convert a batch of samples from this space to a JSONable data type.

### from_jsonable(sample_n)
Convert a JSONable data type to a batch of samples from this space.

## Box

A (possibly unbounded) box in R^n. Specifically, a Box represents the Cartesian product 
of n closed intervals. Each interval has the form of one of [a, b], (-oo, b], [a, oo), 
or (-oo, oo).

There are two common use cases:

* Identical bound for each dimension:
```python
    >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
    Box(3, 4)
```
* Independent bound for each dimension:
```python
    >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
    Box(2,)
```

### is_bounded(manner='both')
Returns a boolean array that indicates whether each element of the box is bounded. The 
returned array has the same shape as the space. Accepts an optional `manner` argument 
which specifies the direction that the element is bounded. It has the following 
options: `below`, `above`, `both`.

### sample()

Generates a single random sample inside of the Box.

In creating a sample of the box, each coordinate is sampled according to the form of the interval:

- [a, b] : uniform distribution
- [a, oo) : shifted exponential distribution
- (-oo, b] : shifted negative exponential distribution
- (-oo, oo) : normal distribution

## Discrete

A single discrete value in $\{ 0, 1, ..., n-1 \}$. A start value can be optionally 
specified to shift the range to $\{ a, a+1, ..., a+n-1 \}$.

Example:

```python
>>> Discrete(2)
>>> Discrete(3, start=-1)  # {-1, 0, 1}
```

## MultiBinary

An n-shape binary space.

The argument to MultiBinary defines n, which could be a number or a list of numbers.

Example Usage:

```python
>>> self.observation_space = spaces.MultiBinary(5)
>>> self.observation_space.sample()
array([0, 1, 0, 1, 0], dtype=int8)

>>> self.observation_space = spaces.MultiBinary([3, 2])
>>> self.observation_space.sample()
array([[0, 0],
       [0, 1],
       [1, 1]], dtype=int8)
```

## MultiDiscrete

The multi-discrete action space consists of a series of discrete action spaces with different number of actions in each. It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space. It is parametrized by passing an array of positive integers specifying number of actions for each discrete action space.

Note: Some environment wrappers assume a value of 0 always represents the NOOP action.

e.g. The Nintendo Game Controller can be conceptualized as 3 discrete action spaces:

- 1. Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
- 2. Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
- 3. Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

It can be initialized as `MultiDiscrete([ 5, 2, 2 ])`

## Dict

A dictionary of simpler spaces.

Example usage:
```python
self.observation_space = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})
```

Example usage (nested):
```python
self.nested_observation_space = spaces.Dict({
    'sensors':  spaces.Dict({
        'position': spaces.Box(low=-100, high=100, shape=(3,)),
        'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
        'front_cam': spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(10, 10, 3)),
            spaces.Box(low=0, high=1, shape=(10, 10, 3))
        )),
        'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
    }),
    'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
    'inner_state':spaces.Dict({
        'charge': spaces.Discrete(100),
        'system_checks': spaces.MultiBinary(10),
        'job_status': spaces.Dict({
            'task': spaces.Discrete(5),
            'progress': spaces.Box(low=0, high=100, shape=()),
        })
    })
})
```

## Tuple

A tuple (i.e., product) of simpler spaces.

Example usage:
```python
>>> self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
```

## Utility Functions

### flatdim(space)
Return the number of dimensions a flattened equivalent of this space would have.

Accepts a space and returns an integer. Raises ``NotImplementedError`` if the space is 
not defined in ``gym.spaces``.

Example usage:
```
>>> s = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})
>>> spaces.flatdim(s)
5
```

### flatten_space(space)

Flatten a space into a single ``Box``.

This is equivalent to ``flatten()``, but operates on the space itself. The result 
always is a `Box` with flat boundaries. The box has exactly ``flatdim(space)`` 
dimensions. Flattening a sample of the original space has the same effect as taking a
sample of the flattenend space.

Example:

```
>>> box = Box(0.0, 1.0, shape=(3, 4, 5))
>>> box
Box(3, 4, 5)
>>> flatten_space(box)
Box(60,)
>>> flatten(box, box.sample()) in flatten_space(box)
True
```

Example that flattens a discrete space:
```
>>> discrete = Discrete(5)
>>> flatten_space(discrete)
Box(5,)
>>> flatten(box, box.sample()) in flatten_space(box)
True
```

Example that recursively flattens a dict:
```
>>> space = Dict({"position": Discrete(2),
...               "velocity": Box(0, 1, shape=(2, 2))})
>>> flatten_space(space)
Box(6,)
>>> flatten(space, space.sample()) in flatten_space(space)
True
```

### flatten(space, x)

Flatten a data point from a space.

This is useful when e.g. points from spaces must be passed to a neural network, which
only understands flat arrays of floats.

Accepts a space and a point from that space. Always returns a 1D array. Raises 
``NotImplementedError`` if the space is not defined in ``gym.spaces``.

### unflatten(space, x)

Unflatten a data point from a space.

This reverses the transformation applied by ``flatten()``. You must ensure that the 
``space`` argument is the same as for the ``flatten()`` call.

Accepts a space and a flattened point. Returns a point with a structure that matches 
the space. Raises ``NotImplementedError`` if the space is not defined in 
``gym.spaces``.
