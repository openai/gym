import gym
from gym import envs, spaces
import numpy as np
import time
try:
    import cv2   # If no OpenCV installed, that test is ignored
except ImportError:
    cv2 = None
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AtariProcessVideo(gym.Wrapper):
    def __init__(self, env=None):
        super(AtariProcessVideo, self).__init__(env)
        self.setup_frame()

    def setup_frame(self):
        _rgb_to_yiq = np.array([
            [ +0.299, +0.596, +0.211],
            [ +0.587, -0.274, -0.523],
            [ +0.114, -0.322, +0.312],
            ]) / 255.0
        def pf_any(frame):
            frame = frame[34:194, 0:160]
            frame = cv2.resize(frame, (84, 84))
            frame = np.dot(frame, _rgb_to_yiq)
            return frame
        self._process_frame = pf_any
        self.observation_space = spaces.Box(-1.0, 1.0, [84, 84, 3])

    def reset(self):
        return self._process_frame(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._process_frame(obs)
        return obs, reward, done, info


def time_random_rollout(env, count):
    t0 = time.time()
    obs = env.reset()
    total_obs = np.array(obs, np.float32)
    for stepi in range(count):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_obs += obs
        if done:
            env.reset()
    total_obs_mean = np.mean(total_obs) / count
    t1 = time.time()
    return '%0.3fs (mean=%0.3f)' % (t1-t0, total_obs_mean)


# Run this with --capture=no to see results
# On an MBP 15" Retina early 2015, I get
# SeaquestNoFrameskip: 0.267s (mean=62.664)
# SeaquestNoFrameskip+process: 0.657s (mean=0.052)
def test_atari_speed():
    logger.info('SeaquestNoFrameskip-v3: %s' % (time_random_rollout(gym.make('SeaquestNoFrameskip-v3'), 1000)))
def test_atari_process_speed():
    if cv2 is not None:
        logger.info('SeaquestNoFrameskip-v3 + process: %s' % (time_random_rollout(AtariProcessVideo(gym.make('SeaquestNoFrameskip-v3')), 1000)))


def main():
    test_atari_speed()
    test_atari_process_speed()

if __name__ == '__main__': main()
