import gym
import logging
import numpy as np
from gym import wrappers


class MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env, N, T, U, lambda_=1.0, noise_mean=0, noise_sde=1, noise_gaussian=True,):

        self.env = env
        self.N = N
        self.T = T
        self.lambda_ = lambda_
        self.noise_mean = noise_mean
        self.noise_sde = noise_sde
        self.U = U
        self.u_init = U[0]
        self.cost_for_N_sample = np.zeros(shape=(self.N))
        # we use position cost so our objective is to minimize cost

        self.env.reset()
        self.x_init = self.env.env.state # start with any random position between -pi to pi

        # use gaussian noise matrix of size (N * T) with mean noise_mean and sde noise_sde
        # TODO: use other noise distribution, here we only assume noise follows a guassian distribution
        self.noise = np.random.normal(loc=self.noise_mean, scale=self.noise_sde, size=(self.N, self.T))

    def _compute_cost_for_sample_n(self, n):
        self.env.env.state = self.x_init
        for t in range(self.T):
            u_t = self.U[t] + self.noise[n, t]
            ob, reward, done, info = self.env.step([u_t])
            # reward from -16 to 0
            self.cost_for_N_sample[n] += -reward

    # make sure that at least one trajectory has non-zero mass/ has low cost
    def _ensure_low_cost(self, cost, min, factor):
        return np.exp(-factor * (cost - min))

    def roll_out(self, iteration):
        for iter in range(iteration):
            for n in range(self.N):
                self._compute_cost_for_sample_n(n)  # total_cost for trajectory n in T steps

            min = np.min(self.cost_for_N_sample)  # minimum cost of all trajectories
            cost_for_N_samples = self._ensure_low_cost(cost=self.cost_for_N_sample, min=min, factor=1 / self.lambda_)

            norm_constant = np.sum(cost_for_N_samples)
            weights = 1 / norm_constant * cost_for_N_samples

            self.U += [np.sum(weights * self.noise[:, t]) for t in range(self.T)]

            logging.info("updated_U: {}".format(self.U))

            self.env.env.state = self.x_init
            obser, r, done, info = self.env.step([self.U[0]])
            logging.critical("action taken at iteration {}: {:.2f} cost received: {:.2f}".format(iter, self.U[0], -r))
            self.env.render()

            # update U
            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = 0
            # reset cost for the next iteration
            self.cost_for_N_sample[:] = 0
            # save the current state
            self.x_init = self.env.env.state


if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 30
    N_SAMPLES = 1000
    ACTION_MIN = -2.0  # pendulum joint effort in (-2, +2)
    ACTION_MAX = 2.0

    noise_mean = 0
    noise_sde = 10
    lambda_ = 1
    iteration = 100

    # generate random U which is the input signal for T = 0 .. 29
    U = np.random.uniform(low=ACTION_MIN, high=ACTION_MAX, size=TIMESTEPS)

    env = gym.make(ENV_NAME)

    logging.basicConfig(filename="pendulum_mppi_action_cost.log", filemode='w', format='%(levelname)s - %(message)s', level=logging.CRITICAL)

    # logging the setup
    info = {}
    info["N_Samples"] = N_SAMPLES
    info["TIMESTEPS"] = TIMESTEPS
    info["lambda"] = lambda_
    info["noise_mean"] = noise_mean
    info["noise_sde"] = noise_sde
    info["N_ITERATION"] = iteration
    info["initial_U"] = U

    logging.critical("meta_parameters for pendulum mppi solver: {}".format(info))

    mppi_gym = MPPI(env=env, N=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mean=noise_mean, noise_sde=noise_sde)
    mppi_gym.roll_out(iteration=iteration)
