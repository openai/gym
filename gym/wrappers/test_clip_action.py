import gym
from gym.wrappers import ClipAction


def test_clip_action():
    # mountaincar: action-based rewards
    env = gym.make('MountainCarContinuous-v0')
    clipped_env = ClipAction(env)

    env.reset()
    clipped_env.reset()

    action = [10000.]

    _, reward, _, _ = env.step(action)
    _, clipped_reward, _, _ = clipped_env.step(action)

    assert abs(clipped_reward) < abs(reward)
