from typing import Any, Callable, List, Optional

NO_RETURNS_RENDER = [None, "human"]
SINGLE_RENDER = ["single_rgb_array"]


class Renderer:
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
