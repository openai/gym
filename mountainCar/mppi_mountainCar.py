import gym
import logging
import numpy as np

class MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env, N, T, U, lambda_=1.0, noise_mean=0, noise_sigma=1,):

        self.env = env
        self.N = N
        self.T = T
        self.lambda_ = lambda_
        self.noise_mean = noise_mean
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = U[0]
        self.cost_for_N_sample = np.zeros(shape=(self.N))
        # we use position cost so our objective is to minimize cost

        self.env.reset()
        self.x_init = self.env.env.state # start with any random position between [-0.05 to 0.05]

        # use gaussian noise matrix of size (N * T) with mean noise_mean and sde noise_sigma
        # TODO: use other noise distribution, here we only assume noise follows a guassian distribution
        self.noise = np.random.normal(loc=self.noise_mean, scale=self.noise_sigma, size=(self.N, self.T))

    def _compute_cost_for_sample_n(self, n):

        for t in range(self.T):
            u_value = self.U[t] + self.noise[n, t]
            u_t = 0 if u_value <= 0.5 else (1 if u_value <= 1.5 else 2)
            ob, reward, done, info = self.env.step(u_t)
            self.cost_for_N_sample[n] += reward


    # make sure that at least one trajectory has non-zero mass/ has low cost
    def _ensure_low_cost(self, cost, min, factor):
        return np.exp(-factor * (cost - min))

    def roll_out(self, iteration):
        total_reward = 0

        for iter in range(iteration):
            saved_state = self.env.env.state
            for n in range(self.N):
                self.env.env.state = saved_state
                self._compute_cost_for_sample_n(n)  # total_cost for trajectory n in T steps

            self.env.env.state = saved_state
            min = np.min(self.cost_for_N_sample)  # minimum cost of all trajectories
            cost_for_N_samples = self._ensure_low_cost(cost=self.cost_for_N_sample, min=min, factor=1 / self.lambda_)

            norm_constant = np.sum(cost_for_N_samples)
            weights = 1 / norm_constant * cost_for_N_samples

            temp_U = self.U + [np.sum(weights * self.noise[:, t]) for t in range(self.T)]
            for i in range(TIMESTEPS):
                self.U[i] = 0 if temp_U[i] <= 0.5 else (1 if temp_U[i] <= 1.5 else 2)

            logging.info("updated_U: {}".format(self.U))

            self.env.env.state = self.x_init
            obser, r, done, info = self.env.step(self.U[0])

            logging.critical("action taken at iteration {}: {:.2f} cost received: {:.2f}".format(iter, self.U[0], r))
            self.env.render()

            # update U
            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = 1
            # reset cost for the next iteration
            self.cost_for_N_sample[:] = 0
            # save the current state
            self.x_init = self.env.env.state

            total_reward += r
        logging.critical("total reward is {} over 200 iterations".format(total_reward))


if __name__ == "__main__":
    ENV_NAME = "MountainCar-v0"
    TIMESTEPS = 20
    N_SAMPLES = 1000
    ACTION_MIN = 0  # push cart to the left
    ACTION_MAX = 2 # push cart to the right

    noise_mean = 0
    noise_sigma = 10
    lambda_ = 1
    iteration = 200

    # generate random U which is the input signal for T = 0 .. 19
    U = np.random.randint(low=ACTION_MIN, high=ACTION_MAX+1, size=TIMESTEPS)

    env = gym.make(ENV_NAME)

    logging.basicConfig(filename="mountainCar_mppi_action_cost_U.log", filemode='w', format='%(levelname)s - %(message)s', level=logging.INFO)

    # logging the setup
    info = {}
    info["N_Samples"] = N_SAMPLES
    info["TIMESTEPS"] = TIMESTEPS
    info["lambda"] = lambda_
    info["noise_mean"] = noise_mean
    info["noise_sigma"] = noise_sigma
    info["N_ITERATION"] = iteration
    info["initial_U"] = U

    logging.critical("meta_parameters for pendulum mppi solver: {}".format(info))

    mppi_gym = MPPI(env=env, N=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mean=noise_mean, noise_sigma=noise_sigma)
    mppi_gym.roll_out(iteration=iteration)
