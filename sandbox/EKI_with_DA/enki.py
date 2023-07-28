import numpy as np
from pdb import set_trace as bp

# NOTES:
#
# Each iteration automatically computes the error, ||y - g(u)||_cov.
# It is up to the user to check whether convergence has been reached
# through self.converge. Here, "convergence" is defined by default to
# be when 3 iterations follow the newest minimum error. This criterion
# can be changed on initialization by the input "counter_max".

def trunc_svd(X, eps=1.0e-6):
    U, D, V_t = np.linalg.svd(X, full_matrices=False)
    nrank = np.searchsorted(-D, -D[0]*eps, side= "right")
    print("trunc_svd! :", nrank, D.shape,  D)
    return U[:,:nrank], D[:nrank], V_t[:nrank,:]


class EKI:

    # INPUTS:
    # parameters.shape = (num_ensembles, num_parameters)
    # observations = truth + N(0, cov)
    # prior is a Gaussian
    def __init__(self, parameters, truth, cov, filter_type = "EAKF", impose_prior = False, prior_mean = None , prior_cov = None , r = 1.0, counter_max = 3, dt = 0.5):

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
        
        self.filter_type = filter_type
        # Prior information
        self.impose_prior = impose_prior
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        
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
        
        # Step size
        assert (0.0 < dt and dt < 1.0)
        self.dt = dt

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
        error = diff.dot( diff )
        # normalize error
        norm = self.g_t.dot( self.g_t )
        
        print("g_t = ", self.g_t)
        print("g_t_pred = ", self.g[-1].mean(0))
        print("diff = ", diff)
        
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

    
    


    def update_ensemble_prediction(self):
        if not self.impose_prior:
            return np.copy(self.u[-1])
        
        
        u = np.copy(self.u[-1])
    
        # Sizes of u
        us = u[0].size
        # Ensemble size
        J = self.J
        # Means
        u_bar = np.zeros(us)
        for j in range(J):
            u_bar += u[j]
        u_bar = u_bar / J
        
        
        g_t = self.g_t
        cov = self.cov
        
        # Prediction step
        u_p = np.copy(u)
        for j in range(J):
            u_p[j] = u_bar + np.sqrt(1.0/(1 - self.dt))*(u[j] - u_bar)
        return u_p
        
      


    def update_ensemble_analysis(self, u, g):
        # Ensemble size
        J = self.J
        self.g = np.append(self.g, [g], axis=0)
        
        if self.impose_prior:
            g = [np.hstack((g[j], u[j])) for j in range(J)]
            g_t = np.hstack((self.g_t, self.prior_mean))
            cov = np.block([[self.cov, np.zeros((self.cov.shape[0], self.prior_cov.shape[1]))], [np.zeros((self.prior_cov.shape[0], self.cov.shape[1])), self.prior_cov]]) / self.dt
        else:
            g_t = self.g_t
            cov = self.cov
            
            
        # Sizes of u and p
        us = u[0].size
        ps = g[0].size
        
        # means 
        u_bar = np.zeros(us)
        p_bar = np.zeros(ps)
        # Loop through ensemble to start computing means 
        # (all the summations only)
        for j in range(J):
            u_bar += u[j]
            p_bar += g[j]
        # Finalize means 
        u_bar = u_bar / J
        p_bar = p_bar / J
        
        if self.filter_type == "ENKF":
            c_uu = np.zeros((us, us))
            c_up = np.zeros((us, ps))
            c_pp = np.zeros((ps, ps))

            # Loop through ensemble to start computing means and covariances
            # (all the summations only)
            for j in range(J):
                u_hat = u[j]
                p_hat = g[j]
                # Covariance matrices
                c_uu += np.tensordot(u_hat, u_hat, axes=0)
                c_up += np.tensordot(u_hat, p_hat, axes=0)
                c_pp += np.tensordot(p_hat, p_hat, axes=0)

            # Finalize means and covariances
            # (divide by J, subtract of means from covariance sum terms)
            c_uu  = c_uu  / J - np.tensordot(u_bar, u_bar, axes=0)
            c_up  = c_up  / J - np.tensordot(u_bar, p_bar, axes=0)
            c_pp  = c_pp  / J - np.tensordot(p_bar, p_bar, axes=0)



            noise = np.array([np.random.multivariate_normal(np.zeros(ps), cov) for _ in range(J)])
            y = g_t + noise
            tmp = np.linalg.solve(c_pp + cov, np.transpose(y-g))
            u += c_up.dot(tmp).T
        
        
        elif self.filter_type == "EAKF":
            # construct square root matrix
            Z_t = np.zeros((J, us))
            Y_t = np.zeros((J, ps))
            for j in range(J):
                Z_t[j, :] = (u[j] - u_bar) / np.sqrt(J)
                Y_t[j, :] = (g[j] - p_bar) / np.sqrt(J)

            X = Y_t.dot(np.linalg.solve(cov, Y_t.T))
            F, Gamma, _ = np.linalg.svd(X, full_matrices=False)
            temp = F.T.dot(Y_t.dot(np.linalg.solve(cov, g_t - p_bar)))
            u_bar_n = u_bar + Z_t.T.dot(  F.dot( temp / (Gamma + 1.0) ))
            P, sqrt_hat_D, V_t =  trunc_svd(Z_t.T) 
            
            
            Y = V_t.dot(F.dot(F.T.dot(V_t.T)/(Gamma + 1.0)[:, np.newaxis]))
            U, D, _ =  np.linalg.svd(Y, full_matrices=False)
            A = (P.dot(sqrt_hat_D[:, np.newaxis]*U)).dot((np.sqrt(D)/sqrt_hat_D)[:, np.newaxis] * P.T)
            for j in range(J):
                u[j] = u_bar_n + A.dot( u[j] - u_bar) 

        print(u_bar)
        # print(c_uu.diagonal())
        # Store parameters and observations
        self.u = np.append(self.u, [u], axis=0)
        
        # Compute error
        self.compute_error()

        
        
        
     