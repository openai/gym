import pytest
import numpy as np

from gym.envs.toy_text.frozen_lake import generate_random_map

# Test that FrozenLake map generation creates valid maps of various sizes.
def test_frozenlake_dfs_map_generation():

    def frozenlake_dfs_path_exists(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] not in '#H'):
                        frontier.append((r_new, c_new))
        return False

    map_sizes = [5, 10, 200]
    for size in map_sizes:
        new_frozenlake = generate_random_map(size)
        assert len(new_frozenlake) == size
        assert len(new_frozenlake[0]) == size
        assert frozenlake_dfs_path_exists(new_frozenlake)
