""" IDEA: Dict space that supports .getattr """
import dataclasses
from inspect import isclass
from collections import OrderedDict
from collections.abc import Mapping as MappingABC
import typing
from typing import (Any, Dict, Generic, Iterable, KeysView, List, Mapping,
                    Sequence, Tuple, Type, TypeVar, Union, get_type_hints, ValuesView, AbstractSet)
from copy import deepcopy
import numpy as np
import gym
from gym import Space, spaces
from gym.vector.utils import batch_space, concatenate


M = TypeVar("M", bound=Mapping[str, Any])
S = TypeVar("S")
Dataclass = TypeVar("Dataclass")


class TypedDictSpace(spaces.Dict, Mapping[str, Space], Generic[M]):
    """ Subclass of `spaces.Dict` that adds the following:
    - allows custom dtypes other than `dict`;
    - uses type annotations on the class to set default and required items (à-la
      `typing.TypedDict`)
    - allows indexing using attributes access (`space.x` := `space["x"]`)

    ## Examples:

    - Using it just like a regular spaces.Dict:

    >>> from gym.spaces import Box
    >>> s = TypedDictSpace(x=Box(0, 1, (4,), dtype=np.float64))
    >>> s
    TypedDictSpace(x:Box(0.0, 1.0, (4,), float64))
    >>> s.seed(123)
    >>> s.sample()
    {'x': array([0.70787616, 0.3698764 , 0.29010696, 0.10647454])}

    - Using it like a TypedDict: (This equivalent to the above)

    >>> class VisionSpace(TypedDictSpace):
    ...     x: Box = Box(0, 1, (4,), dtype=np.float64)  
    >>> s = VisionSpace()
    >>> s
    VisionSpace(x:Box(0.0, 1.0, (4,), float64))
    >>> s.seed(123)
    >>> s.sample()
    {'x': array([0.70787616, 0.3698764 , 0.29010696, 0.10647454])}
    
    - You can also overwrite the values from the type annotations by passing them to the
      constructor:

    >>> s = VisionSpace(x=spaces.Box(0, 2, (3,), dtype=np.int64))
    >>> s
    VisionSpace(x:Box(0, 2, (3,), int64))
    >>> s.seed(123)
    >>> s.sample()
    {'x': array([2, 1, 0])}

    ### Using custom dtypes
    
    Can use any type here, as long as it can receive the samples from each space as
    keyword arguments.

    One good example of this is to use a `dataclass` as the custom dtype.
    You are strongly encouraged to use a dtype that inherits from the `Mapping` class
    from `collections.abc`, so that samples form your space can be handled similarly to
    regular dictionaries.

    >>> from collections import OrderedDict
    >>> s = TypedDictSpace(x=spaces.Box(0, 1, (4,), dtype=int), dtype=OrderedDict)
    >>> s
    TypedDictSpace(x:Box(0, 1, (4,), int64), dtype=<class 'collections.OrderedDict'>)
    >>> s.seed(123)
    >>> s.sample()
    OrderedDict([('x', array([1, 0, 0, 0]))])

    ### Required items:
    
    If an annotation on the class doesn't have a default value, then it is treated as a
    required argument:
    
    >>> class FooSpace(TypedDictSpace):
    ...     a: spaces.Box = spaces.Box(0, 1, (4,), int)
    ...     b: spaces.Discrete
    >>> s = FooSpace()  # doesn't work!
    Traceback (most recent call last):
      ...
    TypeError: Space of type <class 'sequoia.common.spaces.typed_dict.FooSpace'> requires a 'b' item!
    >>> s = FooSpace(b=spaces.Discrete(5))
    >>> s
    FooSpace(a:Box(0, 1, (4,), int64), b:Discrete(5))
    
    NOTE: spaces can also inherit from each other!
    
    >>> class ImageSegmentationSpace(VisionSpace):
    ...     bounding_box: Box
    ... 
    >>> s = ImageSegmentationSpace(
    ...     x=spaces.Box(0, 1, (32, 32, 3), dtype=float),
    ...     bounding_box=spaces.Box(0, 32, (4, 2), dtype=int),
    ... )
    >>> s
    ImageSegmentationSpace(x:Box(0.0, 1.0, (32, 32, 3), float64), bounding_box:Box(0, 32, (4, 2), int64))
    """

    def __init__(
        self, spaces: Mapping[str, Space] = None, dtype: Type[M] = dict, **spaces_kwargs
    ):
        """Creates the TypedDict space.
        
        Can either pass a dict of spaces, or pass the spaces as keyword arguments.

        Parameters
        ----------
        spaces : Mapping[str, Space], optional
            Dictionary mapping from strings to spaces, by default None
        dtype : Type[M], optional
            Type of outputs to return. By default `dict`, but this can also use any
            other dtype which will accept the values from each space as keyword
            arguments. 

        Raises
        ------
        RuntimeError
            If both `spaces` and **kwargs are used.
        TypeError
            If the class has a type annotation for a space without a value set, and
            that space isn't passed as an argument (either in `spaces` or `**kwargs`).
            This is meant to emulate a required argument.
        """
        # Create the basic `spaces` dict using the type hints on the class, just like an
        # actual TypedDict!
        if spaces and spaces_kwargs:
            raise RuntimeError(f"Can only use one of `spaces` or **kwargs, not both.")
        spaces_from_args = spaces or spaces_kwargs

        # have to use OrderedDict just in case python <= 3.6.x
        spaces_from_annotations: Dict[str, gym.Space] = OrderedDict()
        cls = type(self)
        cls_type_annotations: Dict[str, Type] = get_type_hints(cls)
        if cls_type_annotations:
            for attribute, type_annotation in cls_type_annotations.items():
                # TODO: Can't distinguish ClassVars at the moment unless we import
                # _is_classvar from dataclasses.
                # if _is_classvar(type_annotation, typing=typing):
                #     continue
                # NOTE: emulate a 'required argument' when there is a type
                # annotation, but no value.
                # TODO: How about a None value?
                if isclass(type_annotation) and issubclass(type_annotation, gym.Space):
                    _missing = object()
                    value = getattr(cls, attribute, _missing)                    
                    if value is _missing and attribute not in spaces_from_args:
                        raise TypeError(
                            f"Space of type {type(self)} requires a '{attribute}' item!"
                        )
                    if isinstance(value, gym.Space):
                        # Shouldn't be able to have two annotations with the same name.
                        assert attribute not in spaces_from_annotations
                        # TODO: Should copy the space, so that modifying the class
                        # attribute doesn't affect the instances of that space.
                        spaces_from_annotations[attribute] = deepcopy(value)

        # Avoid the annoying sorting of keys that `spaces.Dict` does if we pass a
        # regular dict.
        spaces = OrderedDict()  # Need to use this for 3.6.x
        spaces.update(spaces_from_annotations)
        spaces.update(spaces_from_args)  # Arguments overwrite the spaces from the annotations.

        if not spaces:
            raise TypeError(
                f"Need to either have type annotations on the class, or pass some "
                f"arguments to the constructor!"
            )
        assert all(isinstance(s, gym.Space) for s in spaces.values()), spaces

        super().__init__(spaces=spaces)
        self.dtype = dtype

    def keys(self) -> AbstractSet[str]:
        return self.spaces.keys()

    def items(self) -> AbstractSet[Tuple[str, Space]]:
        return self.spaces.items()

    def values(self) -> ValuesView[Space]:
        return self.spaces.values()

    def sample(self) -> M:
        dict_sample: dict = super().sample()
        # Gets rid of OrderedDict.
        return self.dtype(**dict_sample)

    def __getattr__(self, attr: str) -> Space:
        if attr != "spaces":
            if attr in self.spaces:
                return self.spaces[attr]
        raise AttributeError(f"Space doesn't have attribute {attr}")

    def __getitem__(self, key: Union[str, int]) -> Space:
        if key not in self.spaces:
            if isinstance(key, int):
                # IDEA: Try to get the item at given index in the keys? a bit like a
                # tuple space?
                # return self[list(self.spaces.keys())[key]]
                pass
        return super().__getitem__(key)

    def __len__(self) -> int:
        return len(self.spaces)

    def contains(self, x: Union[M, Mapping[str, Space]]) -> bool:
        # NOTE: Modifying this so that we allow samples with more values, as long as it
        # has all the required keys.
        if not isinstance(x, self.dtype) or not all(k in x for k in self.spaces):
            return False
        for k, space in self.spaces.items():
            if not space.contains(x[k]):
                return False
        return True

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            + ", ".join([f"{k}:{s}" for k, s in self.spaces.items()])
            + (f", dtype={self.dtype}" if self.dtype is not dict else "")
            + ")"
        )

    def __eq__(self, other):
        if isinstance(other, TypedDictSpace) and self.dtype != other.dtype:
            return False
        return super().__eq__(other)


import gym.vector.utils
from gym.vector.utils.shared_memory import \
    read_from_shared_memory as read_from_shared_memory_
from gym.vector.utils.spaces import batch_space
from gym.vector.utils import concatenate


@batch_space.register(TypedDictSpace)
def _batch_typed_dict_space(space: TypedDictSpace, n: int = 1) -> spaces.Dict:
    return type(space)(
        {key: batch_space(subspace, n=n) for (key, subspace) in space.spaces.items()},
        dtype=space.dtype,
    )


@concatenate.register(TypedDictSpace)
def _concatenate_typed_dicts(
    space: TypedDictSpace,
    items: Union[list, tuple],
    out: Union[tuple, dict, np.ndarray],
) -> Dict:
    return space.dtype(
        **{
            key: concatenate(subspace, [item[key] for item in items], out=out[key])
            for (key, subspace) in space.spaces.items()
        }
    )
