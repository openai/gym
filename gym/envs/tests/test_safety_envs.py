import gym


def test_semisuper_true_rewards():
    env = gym.make('SemisuperPendulumNoise-v0')
    env.reset()

    observation, perceived_reward, done, info = env.step(env.action_space.sample())
    true_reward = info['true_reward']

    # The noise in the reward should ensure these are different. If we get spurious errors, we can remove this check
    assert perceived_reward != true_reward
