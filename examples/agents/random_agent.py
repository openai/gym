import logging
import os

import gym

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of outut.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    outdir = '/tmp/random-agent-results'
    gym.upload(outdir, algorithm_id='random')
    raise"hi"

    env = gym.make('CartPole-v0')
    agent = RandomAgent(env.action_space)

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env.monitor.start(outdir, force=True)

    episode_count = 200
    max_steps = 100
    reward = 0
    done = False

    for i in xrange(episode_count):
        ob = env.reset()

        for j in xrange(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

    # Dump result info to disk
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    gym.upload(outdir, algorithm_id='random')
