from typing import Optional
import gym
import numpy as np
import pygame
from pygame.constants import SRCALPHA

OBS_GOAL_IS_NORTH = np.array([0, 1, 1])
OBS_GOAL_IS_SOUTH = np.array([1, 1, 0])
OBS_CORRIDOR = np.array([1, 0, 1])
OBS_T_JUNCTION = np.array([0, 1, 0])
OBS_END_NORTH = np.array([1, 1, 1]) # not specified in the paper
OBS_END_SOUTH = np.array([0, 0, 0]) # not specified in the paper

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

WRONG_ACTION_PENALTY = -1
CORRECT_ACTION_REWARD = 4

class MemoryTMaze(gym.Env):
    """
    The Long-term dependency T-maze with length of corridor N = 10 from
    [Reinforcement Learning with Long Short-Term Memory](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)
    The environment looks like below: at the starting position S the agent's 
    observation indicates where the goal position G is in this episode with X. 

    +---------------------+
    |X                  |G|
    |S| | | | | | | | | | |
    |                   | |
    +---------------------+
    """

    metadata = {"render.modes": ["human", "rgb_array"]}
    def __init__(self, corridor_length = 10) -> None:
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 1, (3,))
        self.corridor_length = corridor_length

        # pygame utils
        self.window_size = (min(64 * corridor_length+1, 512), min(64 * 3, 512))
        self.window_surface = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        choice = self.np_random.choice(2)
        self.position = 0
        if choice == 1:
            self.initilal_state = OBS_GOAL_IS_NORTH
        else:
            self.initilal_state = OBS_GOAL_IS_SOUTH
        if not return_info:
            return np.array(self.initilal_state, dtype=np.float32)
        else:
            return np.array(self.initilal_state, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0
        done = False
        if action == LEFT:
            self.position -= 1
            if self.position < 0:
                reward = WRONG_ACTION_PENALTY
                self.position = 0
        elif action == RIGHT:
            self.position = self.position + 1
            if self.position > self.corridor_length:
                reward = WRONG_ACTION_PENALTY
                self.position = self.corridor_length
        
        if self.position == self.corridor_length:
            self.state = OBS_T_JUNCTION
        elif self.position == 0:
            self.state = self.initilal_state
        else:
            self.state = OBS_CORRIDOR

        if action == DOWN:
            if self.position == self.corridor_length:
                self.state = OBS_END_SOUTH
                done = True
                reward = CORRECT_ACTION_REWARD
            else:
                reward = WRONG_ACTION_PENALTY
        elif action == UP:
            if self.position == self.corridor_length:
                self.state = OBS_END_NORTH
                done = True
                reward = CORRECT_ACTION_REWARD
            else:
                reward = WRONG_ACTION_PENALTY

        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def render(self, mode="human"):
        if self.window_surface is None:
            pygame.init()
            pygame.display.set_caption("Frozen Lake")
            if mode == "human":
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)

        board = pygame.Surface(self.window_size, flags=SRCALPHA)
        cell_width = self.window_size[0] // self.ncol
        cell_height = self.window_size[1] // self.nrow
        smaller_cell_scale = 0.6
        small_cell_w = smaller_cell_scale * cell_width
        small_cell_h = smaller_cell_scale * cell_height