import sys
sys.path.append('../src')
import time
import numpy as np
import multiprocessing
from enki import EKI
import pickle

from L96M_ import L96M
import scipy.integrate as scint
import time
import mkl

np.random.seed(100)

def H(states):
    ## Observe first eight states (mean and covariance)
    mkl.set_num_threads(1)
    obs = np.zeros(44)
    mu = np.mean(states, 0)
    cov = np.cov(states.T)
    obs[:8] = mu
    index = np.triu_indices(8)
    obs[8:] = cov[index]
    return obs

def predictor(x, params):
    ## Error model
    mkl.set_num_threads(1)
    return params[0] * np.tanh(params[1]*x) + params[2] * np.tanh(params[3]*x**2)

def G(u, z0, y):
    ## Forward model G
    mkl.set_num_threads(1)
    l96m = L96M()
    l96m.set_stencil()
    l96m.set_predictor(predictor)
    l96m.set_params(u)
    T = 120 
    dt = 0.01
    t = np.arange(0.0, T, dt)
    sol_reg = scint.solve_ivp(
            l96m.regressed,
            [0,T],
            z0[:l96m.K],
            method = 'LSODA',
            t_eval = t,
            max_step = dt)
    sol_reg = sol_reg.y.T
    return H(sol_reg[int(20.0/dt):,:8])

def ensembleG(ui, z0, y):
    ## Parallel computing of forward model G
    Gmatrix = np.zeros([ui.shape[0],44])
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = []
    for iterN in range(ui.shape[0]):
        results.append(pool.apply_async(G, (ui[iterN,:],z0,y)))
    iterN = 0
    for result in results:
        Gmatrix[iterN,:] = result.get()
        iterN = iterN+1
    pool.close()
    return Gmatrix


if __name__ == "__main__":

    ## Generate an initial condition of Lorenz 96 system with warm up simulation
    l96m = L96M()
    l96m.set_stencil()
    z0 = np.empty(l96m.K + l96m.K * l96m.J)
    dt = 0.01
    T_conv = 3 
    dt_conv = 0.01
    dt_eval = 0.01
    
    z0[:l96m.K] = np.random.rand(l96m.K) * 15 - 5
    for k_ in range(0,l96m.K):
      z0[l96m.K + k_*l96m.J : l96m.K + (k_+1)*l96m.J] = z0[k_]
    
    t_range_conv = np.arange(0, T_conv, dt_eval)
    
    sol_conv = scint.solve_ivp(
            l96m.full,
            [0,T_conv],
            z0,
            method = 'LSODA',
            t_eval = t_range_conv,
            max_step = dt)
    
    z0_conv = sol_conv.y[:,-1]
    del sol_conv
    
    ## Load training data
    y = np.loadtxt('../data/L96_c_10_y_mean.dat')
    y_cov = np.loadtxt('../data/L96_c_10_y_cov.dat')
    y_cov = np.diag(np.diag(y_cov))
    
    ## Set up EKI to calibrate the parameters in error model
    DAsteps = 20 
    numS = 100 
    params = np.ones([numS,4])
    for iterN in range(params.shape[1]):
        params[:,iterN] = np.random.uniform(-1., 1, numS) 
    
    ## Run EKI
    eki = EKI(params, y, y_cov, 1)
    for iterN in range(DAsteps):
        print('DA step: ', iterN+1)
        eki.EnKI(ensembleG(eki.u[iterN], z0_conv, y))
        print("Error: ", eki.error[-1])
        pickle.dump(eki.u, open("L96_c_10_indirect_u.pkl", "wb"))
        pickle.dump(eki.g, open("L96_c_10_indirect_g.pkl", "wb"))
