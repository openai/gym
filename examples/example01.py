# -*- coding: utf-8 -*-
import argparse
import numpy

from gym.agents import LazyEpsilonGreedy
from gym.agents import RandomAgent
from gym.managers import SimpleManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='FrozenLake-v0',
                        help='Select the environment to run')
    args = parser.parse_args()

    manager = SimpleManager(args.env_id)
    agent = RandomAgent(manager.get_action_space())
    r1 = manager.run(agent, episode_count=10)

    manager = SimpleManager(args.env_id)
    agent = LazyEpsilonGreedy(manager.get_action_space(), epsilon=0.2)
    r2 = manager.run(agent, episode_count=10)

    print('\n===== SUMMARY =====')
    for agent_name, values in r1.items():
        print(agent_name)
        print(f'avg actions: {numpy.mean(values["actions"])}')
        print(f'avg reward: {numpy.mean(values["rewards"])}')

    for agent_name, values in r2.items():
        print(agent_name)
        print(f'avg actions: {numpy.mean(values["actions"])}')
        print(f'avg reward: {numpy.mean(values["rewards"])}')
