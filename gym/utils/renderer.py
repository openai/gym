from typing import Any, Callable, List, Optional

# list of modes with which render function returns None
NO_RETURNS_RENDER = [None, "human"]

# list of modes with which render returns just a single frame of the current state
SINGLE_RENDER = ["single_rgb_array"]


class Renderer:
    """ This class serves to easily integrate collection of renders for environments
    that has a function that computes a single render.

    To use this function:
    - instantiate this class with the mode and the function that computes a single frame
    - call render_step method each time the frame should be saved in the list
      (usually at the end of the step method)
    - call get_renders whenever you want to retrieve renders
      (usually in the render method)
    - call reset to clean the render list
      (usually in the reset method of the environment)
    """

    def __init__(self, mode: Optional[str], render: Callable[[str], Any]):
        self.mode = mode
        self.render = render
        self.render_list = []

    def render_step(self) -> None:
        if self.mode is not None and self.mode not in SINGLE_RENDER:
            render_return = self.render(self.mode)
            if self.mode not in NO_RETURNS_RENDER:
                self.render_list.append(render_return)

    def get_renders(self) -> Optional[List]:
        if self.mode in SINGLE_RENDER:
            return self.render(self.mode)
        elif self.mode not in NO_RETURNS_RENDER:
            renders = self.render_list
            self.render_list = []
            return renders

    def reset(self):
        self.render_list = []
