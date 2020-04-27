# -*- coding: utf-8 -*-
import argparse
import numpy

from gym.agents import RandomAgent
from gym.agents import TicTacToeStupidSolver
from gym.managers import OneVsOne


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='TicTacToe-v0',
                        help='Select the environment to run')
    args = parser.parse_args()

    manager = OneVsOne(args.env_id)
    agent1 = RandomAgent(manager.get_action_space())  # whoami = 0
    agent2 = TicTacToeStupidSolver(manager.get_action_space(), whoami=1)
    results = manager.run(agent1, agent2, render=True, episode_count=1)

    print('\n===== SUMMARY =====')
    for agent_name, values in results.items():
        print(agent_name)
        print(f'avg actions: {numpy.mean(values["actions"])}')
        print(f'avg reward: {numpy.mean(values["rewards"])}')

