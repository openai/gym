# -*- coding: utf-8 -*-
import argparse

from gym.agents import RandomAgent
from gym.managers import SimpleManager
from gym.wrappers import Monitor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0',
                        help='Select the environment to run')
    args = parser.parse_args()

    manager = SimpleManager(args.env_id, wrapper=Monitor, wrapper_params={
        'directory': '/tmp/random-agent-results', 'force': True})
    agent = RandomAgent(manager.get_action_space())
    results = manager.run(agent, episode_count=100)
    print(results)
