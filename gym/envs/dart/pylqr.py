# code originally from https://github.com/flforget/DDP

import numpy as np

class PyLQR_iLQRSolver:
    """
    Discrete time finite horizon iLQR solver
    """

    def __init__(self, T, plant_dyn, cost, constraints=None):
        """
        T:              Length of horizon
        plant_dyn:      Discrete time plant dynamics, can be nonlinear
        cost:           instaneous cost function; the terminal cost can be defined by judging the time index
        constraints:    constraints on state/control; will be incorporated into cost

        All the functions should accept (x, u, t, aux) but not necessarily depend on all of them.
        aux indicates the auxiliary arguments to be evaluated in the functions
        """
        self.T = T
        self.plant_dyn = plant_dyn
        self.cost = cost
        self.constraints = constraints
        #auxiliary arguments for function evaluations; particularly useful for cost evaluation
        self.aux = None

        """
        Gradient of dynamics and costs with respect to state/control
        Default is none so finite difference/automatic differentiation will be used
        Otherwise the given functions should be again the functions accept (x, u, t, aux)
        Constraints should mean self.constraints(x, u, t, aux) >= 0
        """
        self.plant_dyn_dx = None        #Df/Dx
        self.plant_dyn_du = None        #Df/Du
        self.cost_dx = None             #Dl/Dx
        self.cost_du = None             #Dl/Du
        self.cost_dxx = None            #D2l/Dx2
        self.cost_duu = None            #D2l/Du2
        self.cost_dux = None            #D2l/DuDx

        self.constraints_dx = None      #Dc/Dx
        self.constraints_du = None      #Dc/Du
        self.constraints_dxx = None     #D2c/Dx2
        self.constraints_duu = None     #D2c/Du2
        self.constraints_dux = None     #D2c/DuDx

        self.constraints_lambda = 1000
        self.finite_diff_eps = 1e-5
        self.reg = .1

        #candidate alphas for line search in the directoin of gradients
        #10 line steps
        self.alpha_array = 1.1 ** (-np.arange(5)**2)

        #adapting the regularizer
        self.reg_max = 1000
        self.reg_min = 1e-6
        self.reg_factor = 10
        return

    def evaluate_trajectory_cost(self, x_array, u_array):
        #Note x_array contains X_T, so a dummy u is required to make the arrays
        #be of consistent length
        u_array_sup = np.vstack([u_array, np.zeros(len(u_array[0]))])

        J_array = [self.cost(x, u, t, self.aux) for t, (x, u) in enumerate(zip(x_array, u_array_sup))]

        return np.sum(J_array)

    def ilqr_iterate(self, x0, u_init, n_itrs=50, tol=1e-6, verbose=True):
        #initialize the regularization term
        self.reg = 1

        #derive the initial guess trajectory from the initial guess of u
        x_array = self.forward_propagation(x0, u_init)

        u_array = np.copy(u_init)
        #initialize current trajectory cost
        J_opt = self.evaluate_trajectory_cost(x_array, u_init)
        J_hist = [J_opt]

        #iterates...
        converged = False
        for i in range(n_itrs):
            k_array, K_array = self.back_propagation(x_array, u_array)

            norm_k = np.mean(np.linalg.norm(k_array, axis=1))
            #apply the control to update the trajectory by trying different alpha
            accept = False
            for alpha in self.alpha_array:
                x_array_new, u_array_new = self.apply_control(x_array, u_array, k_array, K_array, alpha)
                #evaluate the cost of this trial
                J_new = self.evaluate_trajectory_cost(x_array_new, u_array_new)


                if J_new < J_opt:
                    #see if it is converged
                    if np.abs((J_opt - J_new )/J_opt) < tol:
                        #replacement for the next iteration
                        J_opt = J_new
                        x_array = x_array_new
                        u_array = u_array_new
                        converged = True
                        break
                    else:
                        #replacement for the next iteration
                        J_opt = J_new
                        x_array = x_array_new
                        u_array = u_array_new
                        #successful step, decrease the regularization term
                        #momentum like adaptive regularization
                        self.reg = np.max([self.reg_min, self.reg / self.reg_factor])
                        accept = True
                        if verbose:
                            print('Iteration {0}:\tJ = {1};\tnorm_k = {2};\treg = {3}'.format(i+1, J_opt, norm_k, np.log10(self.reg)))
                        break
                else:
                    #don't accept this
                    accept = False

            J_hist.append(J_opt)

            #exit if converged...
            if converged:
                if verbose:
                    print('Converged at iteration {0}; J = {1}; reg = {2}'.format(i+1, J_opt, self.reg))
                break

            #see if all the trials are rejected
            '''if not accept:
                #need to increase regularization
                #check if the regularization term is too large
                if self.reg > self.reg_max:
                    if verbose:
                        print('Exceeds regularization limit at iteration {0}; terminate the iterations'.format(i+1))
                    break

                self.reg = self.reg * self.reg_factor
                if verbose:
                    print('Reject the control perturbation. Increase the regularization term.')'''


        #prepare result dictionary
        res_dict = {
        'J_hist':np.array(J_hist),
        'x_array_opt':np.array(x_array),
        'u_array_opt':np.array(u_array),
        'k_array_opt':np.array(k_array),
        'K_array_opt':np.array(K_array)
        }

        return res_dict

    def apply_control(self, x_array, u_array, k_array, K_array, alpha):
        """
        apply the derived control to the error system to derive new x and u arrays
        """
        x_new_array = [None] * len(x_array)
        u_new_array = [None] * len(u_array)

        x_new_array[0] = x_array[0]
        for t in range(self.T):
            u_new_array[t] = u_array[t] + alpha * (k_array[t] + K_array[t].dot(x_new_array[t] - x_array[t]))
            x_new_array[t+1] = self.plant_dyn(x_new_array[t], u_new_array[t], t, self.aux)

        return np.array(x_new_array), np.array(u_new_array)

    def forward_propagation(self, x0, u_array):
        """
        Apply the forward dynamics to have a trajectory starting from x0 by applying u

        u_array is an array of control signal to apply
        """
        traj_array = [x0]

        for t, u in enumerate(u_array):
            traj_array.append(self.plant_dyn(traj_array[-1], u, t, self.aux))

        return traj_array

    def back_propagation(self, x_array, u_array):
        """
        Back propagation along the given state and control trajectories to solve
        the Riccati equations for the error system (delta_x, delta_u, t)
        Need to approximate the dynamics/costs/constraints along the given trajectory
        dynamics needs a time-varying first-order approximation
        costs and constraints need time-varying second-order approximation
        """
        #Note x_array contains X_T, so a dummy u is required to make the arrays
        #be of consistent length
        u_array_sup = np.vstack([u_array, np.zeros(len(u_array[0]))])
        lqr_sys = self.build_lqr_system(x_array, u_array_sup)

        #k and K
        fdfwd = [None] * self.T
        fdbck_gain = [None] * self.T


        #initialize with the terminal cost parameters to prepare the backpropagation
        Vxx = lqr_sys['dldxx'][-1]
        Vx = lqr_sys['dldx'][-1]


        for t in reversed(range(self.T)):
            #note the double check if we need the transpose or not
            Qx = lqr_sys['dldx'][t] + lqr_sys['dfdx'][t].T.dot(Vx)
            Qu = lqr_sys['dldu'][t] + lqr_sys['dfdu'][t].T.dot(Vx)
            Qxx = lqr_sys['dldxx'][t] + lqr_sys['dfdx'][t].T.dot(Vxx).dot(lqr_sys['dfdx'][t])
            Qux = lqr_sys['dldux'][t] + lqr_sys['dfdu'][t].T.dot(Vxx).dot(lqr_sys['dfdx'][t])
            Quu = lqr_sys['dlduu'][t] + lqr_sys['dfdu'][t].T.dot(Vxx).dot(lqr_sys['dfdu'][t])


            #use regularized inverse for numerical stability
            inv_Quu = self.regularized_persudo_inverse_(Quu, reg=self.reg)

            #get k and K
            fdfwd[t] = -inv_Quu.dot(Qu)
            fdbck_gain[t] = -inv_Quu.dot(Qux)

            #update value function for the previous time step
            Vxx = Qxx - fdbck_gain[t].T.dot(Quu).dot(fdbck_gain[t])
            Vx = Qx - fdbck_gain[t].T.dot(Quu).dot(fdfwd[t])

        return fdfwd, fdbck_gain

    def build_lqr_system(self, x_array, u_array):
        dfdx_array = []
        dfdu_array = []
        dldx_array = []
        dldu_array = []
        dldxx_array = []
        dldux_array = []
        dlduu_array = []

        for t, (x, u) in enumerate(zip(x_array, u_array)):
            #refresh all the points for potential finite difference
            x1 = None
            x2 = None
            u1 = None
            u2 = None

            #for fx
            if self.plant_dyn_dx is not None:
                #use defined derivative
                dfdx_array.append(self.plant_dyn_dx(x, u, t, self.aux))
            else:
                #use finite difference
                if x1 is None or x2 is None:
                    x1 = np.tile(x, (len(x), 1)) + np.eye(len(x)) * self.finite_diff_eps
                    x2 = np.tile(x, (len(x), 1)) - np.eye(len(x)) * self.finite_diff_eps
                fx1 = np.array([self.plant_dyn(x1_dim, u, t, self.aux) for x1_dim in x1])
                fx2 = np.array([self.plant_dyn(x2_dim, u, t, self.aux) for x2_dim in x2])
                dfdx_array.append((fx1-fx2).T/2./self.finite_diff_eps)

            #for fu
            if self.plant_dyn_du is not None:
                #use defined derivative
                dfdu_array.append(self.plant_dyn_du(x, u, t, self.aux))
            else:
                #use finite difference
                if u1 is None or u2 is None:
                    u1 = np.tile(u, (len(u), 1)) + np.eye(len(u)) * self.finite_diff_eps
                    u2 = np.tile(u, (len(u), 1)) - np.eye(len(u)) * self.finite_diff_eps
                fu1 = np.array([self.plant_dyn(x, u1_dim, t, self.aux) for u1_dim in u1])
                fu2 = np.array([self.plant_dyn(x, u2_dim, t, self.aux) for u2_dim in u2])
                dfdu_array.append((fu1-fu2).T/2./self.finite_diff_eps)

            #for lx
            if self.cost_dx is not None:
                #use defined derivative
                dldx_array.append(self.cost_dx(x, u, t, self.aux))
            else:
                #use finite difference
                if x1 is None or x2 is None:
                    x1 = np.tile(x, (len(x), 1)) + np.eye(len(x)) * self.finite_diff_eps
                    x2 = np.tile(x, (len(x), 1)) - np.eye(len(x)) * self.finite_diff_eps
                cx1 = np.array([self.cost(x1_dim, u, t, self.aux) for x1_dim in x1])
                cx2 = np.array([self.cost(x2_dim, u, t, self.aux) for x2_dim in x2])
                dldx_array.append((cx1-cx2).T/2./self.finite_diff_eps)

            #for lu
            if self.cost_du is not None:
                #use defined derivative
                dldu_array.append(self.cost_du(x, u, t, self.aux))
            else:
                #use finite difference
                if u1 is None or u2 is None:
                    u1 = np.tile(u, (len(u), 1)) + np.eye(len(u)) * self.finite_diff_eps
                    u2 = np.tile(u, (len(u), 1)) - np.eye(len(u)) * self.finite_diff_eps
                cu1 = np.array([self.cost(x, u1_dim, t, self.aux) for u1_dim in u1])
                cu2 = np.array([self.cost(x, u2_dim, t, self.aux) for u2_dim in u2])
                dldu_array.append((cu1-cu2).T/2./self.finite_diff_eps)

            #for lxx
            if self.cost_dxx is not None:
                #use defined derivative
                dldxx_array.append(self.cost_dxx(x, u, t, self.aux))
            else:
                #use finite difference
                # l = self.cost(x, u, t, self.aux)
                # dldxx_array.append(np.array([[(cx1_dim + cx2_dim - 2*l)/(self.finite_diff_eps**2) for cx2_dim in cx2] for cx1_dim in cx1]))
                dldxx_array.append(
                    self.finite_difference_second_order_(
                        lambda x_arg: self.cost(x_arg, u, t, self.aux),
                        x))

            #for luu
            if self.cost_duu is not None:
                #use defined derivative
                dlduu_array.append(self.cost_duu(x, u, t, self.aux))
            else:
                #use finite difference
                # l = self.cost(x, u, t, self.aux)
                # dlduu_array.append(np.array([[(cu1_dim + cu2_dim - 2*l)/(self.finite_diff_eps**2) for cu2_dim in cu2] for cu1_dim in cu1]))
                dlduu_array.append(
                    self.finite_difference_second_order_(
                        lambda u_arg: self.cost(x, u_arg, t, self.aux),
                        u))
            #for lux
            if self.cost_dux is not None:
                #use defined derivative
                dldux_array.append(self.cost_dux(x, u, t, self.aux))
            else:
                #use finite difference
                l = self.cost(x, u, t, self.aux)
                cux1 = np.array([[self.cost(x1_dim, u1_dim, t, self.aux) for x1_dim in x1] for u1_dim in u1])
                cux2 = np.array([[self.cost(x2_dim, u2_dim, t, self.aux) for x2_dim in x2] for u2_dim in u2])
                #partial derivative - a simplified approximation, see wiki on finite difference
                dldux = cux1 + cux2 + \
                        2 * np.tile(l, (len(x), len(u))).T - \
                        np.tile(cx1, (len(u), 1)) - np.tile(cx2, (len(u), 1)) - \
                        np.tile(cu1, (len(x), 1)).T - np.tile(cu2, (len(x), 1)).T

                dldux_array.append(dldux/(2*self.finite_diff_eps**2))
            # print dfdx_array[-1], dfdu_array[-1], dldx_array[-1], dldu_array[-1]
            # print dldxx_array[-1], dlduu_array[-1], dldux_array[-1]
            # raw_input()

            #need to do somthing similar for constraints if they were there
            #to incorporate with the cost functions. Ignore them for now
        lqr_sys = {
            'dfdx':dfdx_array,
            'dfdu':dfdu_array,
            'dldx':dldx_array,
            'dldu':dldu_array,
            'dldxx':dldxx_array,
            'dlduu':dlduu_array,
            'dldux':dldux_array
            }

        return lqr_sys

    def regularized_persudo_inverse_(self, mat, reg=1e-5):
        """
        Use SVD to realize persudo inverse by perturbing the singularity values
        to ensure its positive-definite properties
        """
        u, s, v = np.linalg.svd(mat)
        s[ s < 0 ] = 0.0        #truncate negative values...
        diag_s_inv = np.zeros((v.shape[0], u.shape[1]))
        diag_s_inv[0:len(s), 0:len(s)] = np.diag(1./(s+reg))
        return v.dot(diag_s_inv).dot(u.T)

    def finite_difference_second_order_(self, func, x):
        n_dim = len(x)
        func_x = func(x)

        hessian = np.zeros((n_dim, n_dim))
        for i in range(n_dim):
            for j in range(n_dim):
                x_copy = np.copy(x)
                x_copy[i] += self.finite_diff_eps
                x_copy[j] += self.finite_diff_eps
                fpp = func(x_copy)

                x_copy = np.copy(x)
                x_copy[i] += self.finite_diff_eps
                fp_ = func(x_copy)

                x_copy = np.copy(x)
                x_copy[j] += self.finite_diff_eps
                f_p = func(x_copy)

                x_copy = np.copy(x)
                x_copy[i] -= self.finite_diff_eps
                fn_ = func(x_copy)

                x_copy = np.copy(x)
                x_copy[j] -= self.finite_diff_eps
                f_n = func(x_copy)

                x_copy = np.copy(x)
                x_copy[i] -= self.finite_diff_eps
                x_copy[j] -= self.finite_diff_eps
                fnn = func(x_copy)

                hessian[i, j] = fpp - fp_ - f_p - f_n - fn_ + fnn

        hessian = (hessian + 2*func_x) / (2*self.finite_diff_eps**2)

        return hessian