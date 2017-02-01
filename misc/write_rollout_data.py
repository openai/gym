"""
This script does a few rollouts with an environment and writes the data to an npz file
Its purpose is to help with verifying that you haven't functionally changed an environment.
(If you have, you should bump the version number.)
"""
import argparse, numpy as np, collections, sys
from os import path


class RandomAgent(object):
    def __init__(self, ac_space):
        self.ac_space = ac_space
    def act(self, _):
        return self.ac_space.sample()

def rollout(env, agent, max_episode_steps):
    """
    Simulate the env and agent for max_episode_steps
    """
    ob = env.reset()
    data = collections.defaultdict(list)
    for _ in xrange(max_episode_steps):
        data["observation"].append(ob)
        action = agent.act(ob)
        data["action"].append(action)
        ob,rew,done,_ = env.step(action)
        data["reward"].append(rew)
        if done:
            break
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("envid")
    parser.add_argument("outfile")
    parser.add_argument("--gymdir")

    args = parser.parse_args()
    if args.gymdir:
        sys.path.insert(0, args.gymdir)
    import gym
    from gym import utils
    print utils.colorize("gym directory: %s"%path.dirname(gym.__file__), "yellow")
    env = gym.make(args.envid)
    agent = RandomAgent(env.action_space)
    alldata = {}
    for i in xrange(2):
        np.random.seed(i)
        data = rollout(env, agent, env.spec.max_episode_steps)
        for (k, v) in data.items():
            alldata["%i-%s"%(i, k)] = v
    np.savez(args.outfile, **alldata)

if __name__ == "__main__":
    main()
