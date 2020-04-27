# -*- coding: utf-8 -*-
from gym import logger
from gym import make


class BaseManager(object):
    """ A base manager that takes care of running a given `agent` on a given
    `environment`. It supports also `wrappers`.

    Args:
        env_id: a string id which specify the env
        env_seed: an int for the random seed (default env_seed=0)
        wrapper: a class for an env `wrapper`
        wrapper_params: a dict: {'param1': value1, 'param2': value2, ...} used
                        to create the wrapper instance
    """

    def __init__(self, env_id, env_seed=0, wrapper=None, wrapper_params={}):
        self.env = make(env_id)
        if wrapper:
            self.env = wrapper(self.env, **wrapper_params)
        self.env.seed(env_seed)

    def get_action_space(self):
        return self.env.action_space


class SimpleManager(BaseManager):
    """ A simple manager that takes care of running a given `agent` on a given
    `environment`. It supports also `wrappers`.

    Args:
        env_id: a string id which specify the env
        env_seed: an int for the random seed (default env_seed=0)
        wrapper: a class for an env `wrapper`
        wrapper_params: a dict: {'param1': value1, 'param2': value2, ...} used
                        to create the wrapper instance
    """

    def __init__(self, env_id, env_seed=0, wrapper=None, wrapper_params={}):
        super().__init__(env_id=env_id, env_seed=env_seed,
                         wrapper=wrapper, wrapper_params=wrapper_params)

    def run(self, agent, episode_count, initial_reward=0, render=True,
            render_mode='human'):
        """ Runs `episode_count` on given `agent` and env.

        Args:
            agent: an agent derived from `gym.agent.BaseAgent`
            episode_count: the number of episodes
            initial_reward: the initial reward
            render: boolean if true render with render_mode
            render_mode: one the accepted render modes (commons are: 'human',
                         'ansi', ...)
        Returns:
            a dict: for each episode the total number of actions an total reward
        """
        results = {
            agent.name: {
                'actions': [],
                'rewards': [],
            }}
        for i in range(episode_count):
            logger.info(f'episode: {i}')

            reward = initial_reward
            done = False

            agent.reset()
            ob = self.env.reset()
            while True:
                if render: self.env.render(mode=render_mode)
                action = agent.act(ob, reward, done)
                ob, reward, done, _ = self.env.step(action)
                if done:
                    # this is needed to update the final reward
                    agent.act(ob, reward, done)
                    logger.info(agent)
                    break
            if render: self.env.render(mode=render_mode)
            results[agent.name]['actions'].append(agent.actions)
            results[agent.name]['rewards'].append(agent.reward)
        self.env.close()
        return results


class OneVsOne(BaseManager):
    """ A manager that takes care of running two `agents` alternating on a given
    `environment`. It supports also `wrappers`.

    Args:
        env_id: a string id which specify the env
        env_seed: an int for the random seed (default env_seed=0)
        wrapper: a class for an env `wrapper`
        wrapper_params: a dict: {'param1': value1, 'param2': value2, ...} used
                        to create the wrapper instance
    """

    def __init__(self, env_id, env_seed=0, wrapper=None, wrapper_params={}):
        super().__init__(env_id=env_id, env_seed=env_seed,
                         wrapper=wrapper, wrapper_params=wrapper_params)

    def run(self, agent1, agent2, episode_count, initial_reward=0, render=True,
            render_mode='human'):
        """ Runs `episode_count` on given pair `agent1` `agent2` (alternating
        them) on the selected env.

        Args:
            agent1: an agent derived from `gym.agent.BaseAgent`
            agent2: an agent derived from `gym.agent.BaseAgent`
            episode_count: the number of episodes
            initial_reward: the initial reward
            render: boolean if true render with render_mode
            render_mode: one the accepted render modes (commons are: 'human',
                         'ansi', ...)
        Returns:
            a dict: for each episode the total number of actions an total reward
        """
        results = {
            agent1.name: {
                'actions': [],
                'rewards': [],
            },
            agent2.name: {
                'actions': [],
                'rewards': [],
            }}
        for i in range(episode_count):
            logger.info(f'episode: {i}')

            reward1 = initial_reward
            reward2 = initial_reward
            done = False

            agent1.reset()
            agent2.reset()
            ob = self.env.reset()
            while True:
                if render: self.env.render(mode=render_mode)
                print(agent1.name)
                action = agent1.act(ob, reward1, done)
                ob, reward1, done, _ = self.env.step(action)

                if done:
                    # this is needed to update the final reward
                    agent1.act(ob, reward1, done)
                    logger.info(agent1)
                    logger.info(agent2)
                    break

                if render: self.env.render(mode=render_mode)
                print(agent2.name)
                action = agent2.act(ob, reward2, done)
                ob, reward2, done, _ = self.env.step(action)

                if done:
                    # this is needed to update the final reward
                    agent2.act(ob, reward2, done)
                    logger.info(agent1)
                    logger.info(agent2)
                    break
            if render: self.env.render(mode=render_mode)
            results[agent1.name]['actions'].append(agent1.actions)
            results[agent1.name]['rewards'].append(agent1.reward)
            results[agent2.name]['actions'].append(agent2.actions)
            results[agent2.name]['rewards'].append(agent2.reward)
        self.env.close()
        return results
