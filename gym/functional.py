"""Base class and definitions for an alternative, functional backend for gym envs, particularly suitable for hardware accelerated and otherwise transformed environments."""

from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

import numpy as np

StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")
RewardType = TypeVar("RewardType")
TerminalType = TypeVar("TerminalType")
RenderStateType = TypeVar("RenderStateType")


class FuncEnv(Generic[StateType, ObsType, ActType, RewardType, TerminalType]):
    """Base class (template) for functional envs.

    This API is meant to be used in a stateless manner, with the environment state being passed around explicitly.
    That being said, nothing here prevents users from using the environment statefully, it's just not recommended.
    A functional env consists of three functions (in this case, instance methods):
    - initial: returns the initial state of the POMDP
    - observation: returns the observation in a given state
    - step: returns the next state, reward, and terminated flag after taking an action in a given state

    The class-based structure serves the purpose of allowing environment constants to be defined in the class,
    and then using them by name in the code itself.

    For the moment, this is predominantly for internal use. This API is likely to change, but in the future
    we intend to flesh it out and officially expose it to end users.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """Initialize the environment constants."""
        self.__dict__.update(options or {})

    def initial(self, rng: Any) -> StateType:
        """Initial state."""
        raise NotImplementedError

    def observation(self, state: StateType) -> ObsType:
        """Observation."""
        raise NotImplementedError

    def transition(self, state: StateType, action: ActType, rng: Any) -> StateType:
        """Transition."""
        raise NotImplementedError

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> RewardType:
        """Reward."""
        raise NotImplementedError

    def terminal(self, state: StateType) -> TerminalType:
        """Terminal state."""
        raise NotImplementedError

    def transform(self, func: Callable[[Callable], Callable]):
        """Functional transformations."""
        self.initial = func(self.initial)
        self.transition = func(self.transition)
        self.observation = func(self.observation)
        self.reward = func(self.reward)
        self.terminal = func(self.terminal)

    def render_image(
        self, state: StateType, render_state: RenderStateType
    ) -> Tuple[RenderStateType, np.ndarray]:
        """Show the state."""
        raise NotImplementedError
