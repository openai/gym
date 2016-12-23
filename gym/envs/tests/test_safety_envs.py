import gym


def test_semisuper_true_rewards():
    env = gym.make('SemisuperPendulumNoise-v0')
    env.reset()

    observation, perceived_reward, done, info = env.step(env.action_space.sample())
    true_reward = info['true_reward']

    assert perceived_reward != true_reward

if __name__ == '__main__':
    test_semisuper_true_rewards()
