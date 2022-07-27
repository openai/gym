"""Root __init__ of the gym dev_wrappers."""
from typing import Dict, Sequence, TypeVar, Union

ArgType = TypeVar("ArgType")
FuncArgType = Union[ArgType, Dict[str, "FuncArgType"], Sequence["FuncArgType"]]
