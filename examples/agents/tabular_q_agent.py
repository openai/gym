class TabularQAgent(object):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, discrete.Discrete):
            raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        if not isinstance(action_space, discrete.Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "learning_rate" : 0.1,
            "eps": 0.05,            # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_iter": 10000}        # Number of iterations
        self.config.update(userconfig)
        self.q = defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"])

    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        action = np.argmax(self.q[observation.item()]) if np.random.random() > eps else self.action_space.sample()
        return action

    def learn(self, env):
        config = self.config
        obs = env.reset()
        q = self.q
        for t in range(config["n_iter"]):
            action, _ = self.act(obs)
            obs2, reward, done, _ = env.step(action)
            future = 0.0
            if not done:
                future = np.max(q[obs2.item()])
            q[obs.item()][action] -= \
                self.config["learning_rate"] * (q[obs.item()][action] - reward - config["discount"] * future)

            obs = obs2
