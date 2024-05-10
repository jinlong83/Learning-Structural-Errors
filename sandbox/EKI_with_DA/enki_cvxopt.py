import numpy as np
import cvxopt
from cvxopt import matrix, spdiag
import sys

# NOTES:
#
# Each iteration automatically computes the error, ||y - g(u)||_cov.
# It is up to the user to check whether convergence has been reached
# through self.converge. Here, "convergence" is defined by default to
# be when 3 iterations follow the newest minimum error. This criterion
# can be changed on initialization by the input "counter_max".

class EKI:

    # INPUTS:
    # parameters.shape = (num_ensembles, num_parameters)
    # observations = truth + N(0, cov)
    def __init__(self, parameters, truth, cov, r = 1.0, counter_max = 3):

        assert (parameters.ndim == 2), \
            'EKI init: parameters must be 2d array, num_ensembles x num_parameters'
        assert (truth.ndim == 1), 'EKI init: truth must be 1d array'
        assert (cov.ndim == 2), 'EKI init: covariance must be 2d array'
        assert (truth.size == cov.shape[0] and truth.size == cov.shape[1]),\
            'EKI init: truth and cov are not the correct sizes'

        # Truth statistics
        self.g_t = truth
        self.cov = r**2 * cov
        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.linalg.LinAlgError:
            print('cov not invertible')
            self.cov_inv = np.ones(cov.shape)

        # Parameters
        self.u = parameters[np.newaxis]

        # Ensemble size
        self.J = parameters.shape[0]

        # Store observations during iterations
        self.g = np.empty((0,self.J)+truth.shape)

        # Error
        self.error = np.empty(0)

        # Convergence/minimum error
        self.min_error = None
        self.counter = 0
        self.counter_max = counter_max
        self.converged = False

        # Additional stuff to be used as needed
        self.lat = None

        self.eigmin = 1e-3

        # Sparse EKI settings
        # self.lam = 1.0  # L1-norm upper bound
        # self.sparse_threshold = 0.1  # threshold for sparsity
        # self.inflation_std = 1e-3  # standard deviation for inflation
        self.lam = 1.0  # L1-norm upper bound
        self.sparse_threshold = 0.1  # threshold for sparsity
        self.inflation_std = 1e-3  # standard deviation for inflation (parameters)
        self.cov_inflation_std = 1e-8  # covariance inflation (data and parameters)

    # Parameters corresponding to minimum error.
    # Returns mean and standard deviation.
    def get_u(self):
        try:
            idx = self.error.argmin()
            u = self.u[idx+1]
            return u.mean(0), np.std(u, axis=0)
        except ValueError:
            print('No errors computed.')

    # Minimum error
    def get_error(self):
        try:
            idx = self.error.argmin()
            return self.error[idx]
        except ValueError:
            print('No errors computed.')

    # Compute error and track convergence. Error is only computed for
    # the most recent iteration.
    def compute_error(self):
        diff = self.g_t - self.g[-1].mean(0)
        error = diff.dot(np.linalg.solve(self.cov, diff))
        norm = self.g_t.dot(np.linalg.solve(self.cov, self.g_t))
        error = error/norm

        self.error = np.append(self.error, error)

        if self.min_error == None:
            self.min_error = self.error.min()
            return

        if self.min_error < self.error[-1]:
            self.counter += 1
        else:
            self.min_error = self.error[-1]
            self.counter = 0

        if self.counter > 3:
            self.converged = True

    def getAplus(self, A, threshold):
        eigval, eigvec = np.linalg.eig(A)
        Q = eigvec
        xdiag = np.diag(np.maximum(eigval, threshold))
        return np.real(np.dot(np.dot(Q, xdiag), Q.T))

    def cvxopt_solve_qp(self, Hs, P, q, G=None, h=None, A=None, b=None):
        n = P.shape[0]
        params_n = Hs.shape[0]
        obs_n = n - params_n
        P = .5 * (P + P.T)  # make sure P is symmetric
        P1 = spdiag([matrix(P)] + [0] * params_n)
        # try:
        #    np.linalg.cholesky(P1)
        # except:
        #    P1 = self.getAplus(P1, self.eigmin)
        zeros = matrix([0.] * params_n)
        q1 = matrix([matrix(q), zeros])
        args_qp = [P1, q1]
        if G is not None:
            eye_n = matrix(Hs)
            eye_n_s = spdiag([1.0] * params_n)
            vec_n = matrix([matrix([0.0] * n), matrix([1.0] * params_n)])
            G1 = matrix([ [eye_n, -1 * eye_n], [-1 * eye_n_s, -1 * eye_n_s] ])
            h1 = matrix(0., (2 * params_n, 1) )
            Gnew = matrix([G1, vec_n.T])
            hnew = matrix([h1, self.lam])
            args_qp.extend([Gnew, hnew])
            if A is not None:
                args_qp.extend([matrix(A), matrix(b)])
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(*args_qp)
        if 'optimal' not in sol['status']:
            return None
        return np.array(sol['x']).reshape((n + params_n,))

    def QP(self, w, w_hat, c_ww_inv, H, Hs):
        P = np.matmul(np.matmul(H.T, self.cov_inv), H) + c_ww_inv
        q = -(np.dot(c_ww_inv.T, w_hat) + np.dot(np.matmul(H.T, self.cov_inv.T), self.g_t))
        G = np.zeros(P.shape)
        h = np.zeros(q.shape)

        print("cvxopt_solve_qp")
        print("Hs", Hs)
        print("P", P)
        print("q", q)
        optim = self.cvxopt_solve_qp(Hs,P,q,G,h)
        v_optim = optim[:P.shape[0]]
        return np.ndarray.flatten(v_optim)

    def constrainedOptim(self, w, w_hat, c_ww_inv, Hp, Hs):
        print("constrainedOptim")
        print("w", w)
        print("w_hat", w_hat)
        print("c_ww_inv", c_ww_inv)
        w_new = self.QP(w, w_hat, c_ww_inv, Hp, Hs[:,:])
        return np.dot(Hs, w_new)

    # g: data, i.e. g(u), with shape (num_ensembles, num_elements)
    def EnKI(self, g):

        u = np.copy(self.u[-1])
        g_t = self.g_t
        cov = self.cov

        # Ensemble size
        J = self.J

        # Sizes of u and p
        us = u[0].size
        ps = g[0].size

        ws = us + ps

        # means and covariances
        u_bar = np.zeros(us)
        p_bar = np.zeros(ps)
        c_uu = np.zeros((us, us))
        c_up = np.zeros((us, ps))
        c_pp = np.zeros((ps, ps))

        w_bar = np.zeros(ws)
        c_ww = np.zeros((ws, ws))

        # Loop through ensemble to start computing means and covariances
        # (all the summations only)
        for j in range(J):

            u_hat = u[j]
            p_hat = g[j]
            w_hat = np.hstack([u_hat,p_hat])

            # Means
            u_bar += u_hat
            p_bar += p_hat
            w_bar += w_hat

            # Covariance matrices
            c_uu += np.tensordot(u_hat, u_hat, axes=0)
            c_up += np.tensordot(u_hat, p_hat, axes=0)
            c_pp += np.tensordot(p_hat, p_hat, axes=0)

        # Finalize means and covariances
        # (divide by J, subtract of means from covariance sum terms)
        u_bar = u_bar / J
        p_bar = p_bar / J
        c_uu  = c_uu  / J - np.tensordot(u_bar, u_bar, axes=0)
        c_up  = c_up  / J - np.tensordot(u_bar, p_bar, axes=0)
        c_pp  = c_pp  / J - np.tensordot(p_bar, p_bar, axes=0)

        c_ww  = np.vstack([np.hstack([c_uu,c_up]),np.hstack([c_up.T,c_pp])])

        if np.linalg.matrix_rank(c_ww) == c_ww.shape[0]:
            c_ww_inv = np.linalg.inv(c_ww)
        else:    
            print("c_ww is not invertible! diagonal noise has been added.")
            c_ww_inv = np.linalg.inv(c_ww + self.cov_inflation_std*np.identity(c_ww.shape[0]))
        w_hat = np.hstack([u, g])
        Hs = np.eye(us, us+ps)
        Hp = np.hstack([np.zeros((ps, us)), np.eye(ps)])

        # # Update u
        noise = np.array([np.random.multivariate_normal(np.zeros(ps), cov) for _ in range(J)])
        y = g_t + noise
        tmp = np.linalg.solve(c_pp + cov, np.transpose(y-g))
        w = np.hstack([u, g])
        u += c_up.dot(tmp).T

        # Update results
        for iterN in range(g.shape[0]):
            u[iterN] = self.constrainedOptim(w[iterN], w_hat[iterN], c_ww_inv, Hp, Hs)
            u[iterN] = u[iterN] * (np.abs(u[iterN]) > self.sparse_threshold)
            u[iterN] = u[iterN] + np.random.normal(0,self.inflation_std,u[iterN].shape[0])

        # Store parameters and observations
        self.u = np.append(self.u, [u], axis=0)
        self.g = np.append(self.g, [g], axis=0)

        # Compute error
        self.compute_error()
