from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import pygame
from pygame.constants import SRCALPHA
import numpy as np

from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == "G":
                        return True
                    if res[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        res[0][0] = "S"
        res[-1][-1] = "G"
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(Env):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, desc=None, map_name="4x4", is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b"GH"
            reward = float(newletter == b"G")
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        # pygame utils
        self.window_size = (
            min(64 * ncol, 512),
            min(64 * nrow, 512)
        )
        self.window_surface = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        return int(self.s)

    def render(self, mode="human"):
        desc = self.desc.tolist()
        if mode == "ansi":
            return self._render_text(desc)
        else:
            return self._render_gui(desc)

    def _render_gui(self, desc):
        if self.window_surface is None:
            pygame.init()
            pygame.display.set_caption("Frozen Lake")
            self.window_surface = pygame.display.set_mode(self.window_size)
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.image.load(file_name)
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.image.load(file_name)
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.image.load(file_name)
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.image.load(file_name)
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.image.load(file_name)
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [pygame.image.load(f_name) for f_name in elfs]

        board = pygame.Surface(self.window_size, flags=SRCALPHA)
        cell_width = self.window_size[0] // self.ncol
        cell_height = self.window_size[1] // self.nrow
        smaller_cell_scale = 0.7
        small_cell_w = smaller_cell_scale * cell_width
        small_cell_h = smaller_cell_scale * cell_height

        # prepare images
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]
        elf_scale = min(
            small_cell_w / elf_img.get_width(),
            small_cell_h / elf_img.get_height(),
        )
        elf_dims = (
            elf_img.get_width() * elf_scale,
            elf_img.get_height() * elf_scale,
        )
        elf_img = pygame.transform.scale(elf_img, elf_dims)
        hole_img = pygame.transform.scale(self.hole_img, (cell_width, cell_height))
        cracked_hole_img = pygame.transform.scale(self.cracked_hole_img, (cell_width, cell_height))
        ice_img = pygame.transform.scale(self.ice_img, (cell_width, cell_height))
        goal_img = pygame.transform.scale(self.goal_img, (small_cell_w, small_cell_h))
        start_img = pygame.transform.scale(self.start_img, (small_cell_w, small_cell_h))

        for y in range(self.nrow):
            for x in range(self.ncol):
                rect = (x * cell_width, y * cell_height, cell_width, cell_height)
                if desc[y][x] == b"H":
                    self.window_surface.blit(hole_img, (rect[0], rect[1]))
                elif desc[y][x] == b"G":
                    self.window_surface.blit(ice_img, (rect[0], rect[1]))
                    goal_rect = self._center_small_rect(rect, goal_img.get_size())
                    self.window_surface.blit(goal_img, goal_rect)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(ice_img, (rect[0], rect[1]))
                    stool_rect = self._center_small_rect(rect, start_img.get_size())
                    self.window_surface.blit(start_img, stool_rect)
                else:
                    self.window_surface.blit(ice_img, (rect[0], rect[1]))

                pygame.draw.rect(board, (200, 220, 255), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (
            bot_col * cell_width,
            bot_row * cell_height,
            cell_width,
            cell_height,
        )
        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(cracked_hole_img, (cell_rect[0], cell_rect[1]))
        else:
            elf_rect = self._center_small_rect(cell_rect, elf_img.get_size())
            self.window_surface.blit(elf_img, elf_rect)

        self.window_surface.blit(board, board.get_rect())
        pygame.display.update()

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self, desc):
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()
