"""A utility class to collect render frames from a function that computes a single frame."""
from typing import Any, Callable, List, Optional, Set

# list of modes with which render function returns None
NO_RETURNS_RENDER = {"human"}

# list of modes with which render returns just a single frame of the current state
SINGLE_RENDER = {"single_rgb_array", "single_depth_array", "single_state_pixels"}


class Renderer:
    """This class serves to easily integrate collection of renders for environments that can computes a single render.

    To use this function:
    - instantiate this class with the mode and the function that computes a single frame
    - call render_step method each time the frame should be saved in the list
      (usually at the end of the step and reset methods)
    - call get_renders whenever you want to retrieve renders
      (usually in the render method)
    - call reset to clean the render list
      (usually in the reset method of the environment)
    """

    def __init__(
        self,
        mode: Optional[str],
        render: Callable[[str], Any],
        no_returns_render: Optional[Set[str]] = None,
        single_render: Optional[Set[str]] = None,
    ):
        """Instantiates a Renderer object.

        Args:
            mode (Optional[str]): Way to render
            render (Callable[[str], Any]): Function that receives the mode and computes a single frame
            no_returns_render (Optional[Set[str]]): Set of render modes that don't return any value.
                The default value is the set {"human"}.
            single_render (Optional[Set[str]]): Set of render modes that should return a single frame.
                The default value is the set {"single_rgb_array", "single_depth_array", "single_state_pixels"}.
        """
        if no_returns_render is None:
            no_returns_render = NO_RETURNS_RENDER
        if single_render is None:
            single_render = SINGLE_RENDER

        self.no_returns_render = no_returns_render
        self.single_render = single_render
        self.mode = mode
        self.render = render
        self.render_list = []

    def render_step(self) -> None:
        """Computes a frame and save it to the render collection list.

        This method should be usually called inside environment's step and reset method.
        """
        if self.mode is not None and self.mode not in self.single_render:
            render_return = self.render(self.mode)
            if self.mode not in self.no_returns_render:
                self.render_list.append(render_return)

    def get_renders(self) -> Optional[List]:
        """Pops all the frames from the render collection list.

        This method should be usually called in the environment's render method to retrieve the frames collected till this time step.
        """
        if self.mode in self.single_render:
            return self.render(self.mode)
        elif self.mode is not None and self.mode not in self.no_returns_render:
            renders = self.render_list
            self.render_list = []
            return renders

    def reset(self):
        """Resets the render collection list.

        This method should be usually called inside environment's reset method.
        """
        self.render_list = []
