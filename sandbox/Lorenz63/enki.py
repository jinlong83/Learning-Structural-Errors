import numpy as np

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
        # normalize error
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
        
        # means and covariances
        u_bar = np.zeros(us)
        p_bar = np.zeros(ps)
        c_up = np.zeros((us, ps))
        c_pp = np.zeros((ps, ps))
        
        # Loop through ensemble to start computing means and covariances
        # (all the summations only)
        for j in range(J):
            
            u_hat = u[j]
            p_hat = g[j]
            
            # Means
            u_bar += u_hat
            p_bar += p_hat
            
            # Covariance matrices
            c_up += np.tensordot(u_hat, p_hat, axes=0)
            c_pp += np.tensordot(p_hat, p_hat, axes=0)
            
        # Finalize means and covariances
        # (divide by J, subtract of means from covariance sum terms)
        u_bar = u_bar / J
        p_bar = p_bar / J
        c_up  = c_up  / J - np.tensordot(u_bar, p_bar, axes=0)
        c_pp  = c_pp  / J - np.tensordot(p_bar, p_bar, axes=0)
        
        # # Update u
        noise = np.array([np.random.multivariate_normal(np.zeros(ps), cov) for _ in range(J)])
        y = g_t + noise
        tmp = np.linalg.solve(c_pp + cov, np.transpose(y-g))
        u += c_up.dot(tmp).T

        # Store parameters and observations
        self.u = np.append(self.u, [u], axis=0)
        self.g = np.append(self.g, [g], axis=0)

        # Compute error
        self.compute_error()
