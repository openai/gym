from typing import Sequence, Tuple, Union

_Shape = Tuple[int, ...]

# Anything that can be coerced to a shape tuple
_ShapeLike = Union[int, Sequence[int]]
