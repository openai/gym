import gym
from gym.wrappers import RecordEpisodeStatistics


def test_record_episode_statistics():
    env = gym.make('CartPole-v1')
    env = RecordEpisodeStatistics(env, deque_size=2)

    for n in range(5):
        env.reset()
        assert env.episode_return == 0.0
        assert env.episode_horizon == 0
        for t in range(env.spec.max_episode_steps):
            _, _, done, info = env.step(env.action_space.sample())
            if done:
                assert 'episode' in info
                assert all([item in info['episode'] for item in ['return', 'horizon', 'time']])
                break
    assert len(env.return_queue) == 2
    assert len(env.horizon_queue) == 2
