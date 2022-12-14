import sys
sys.path.append('../src')
import time
import numpy as np
import multiprocessing
from enki import EKI
import pickle

from L96M_ import L96M
import scipy.integrate as scint
import sdeint
from scipy.stats import skew, kurtosis
import time
import mkl

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

np.random.seed(100)

def autocorr(x):
    ## Calculate the autocorrelation function from a time series
    mkl.set_num_threads(1)
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:] / result.max()

def H(states):
    ## Observe all states (first four moments and autocorrelation function)
    mkl.set_num_threads(1)
    obs = np.zeros(153)
    mu = np.mean(states, 0)
    cov = np.var(states, 0)
    obs[:36] = mu
    obs[36:72] = cov
    obs[72:108] = skew(states)
    obs[108:144] = kurtosis(states)
    for i in range(states.shape[1]):
        obs[144:] += autocorr(states[:,i])[30:120:10]
    obs[144:] = obs[144:] / states.shape[1]
    return obs

def predictor(x, params):
    ## Error model (deterministic part)
    mkl.set_num_threads(1)
    hidden_dim = 5
    net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1, 1)
        )
    weights1 = params[:hidden_dim*1].reshape(hidden_dim,1)
    weights1 = torch.nn.parameter.Parameter(torch.tensor(weights1, dtype=torch.float32))
    bias1 = params[hidden_dim*1:hidden_dim*2].reshape(1,hidden_dim)
    bias1 = torch.nn.parameter.Parameter(torch.tensor(bias1, dtype=torch.float32))
    weights2 = params[hidden_dim*2:hidden_dim*3].reshape(1,hidden_dim)
    weights2 = torch.nn.parameter.Parameter(torch.tensor(weights2, dtype=torch.float32))
    bias2 = params[hidden_dim*3:hidden_dim*3+1].reshape(1,1)
    bias2 = torch.nn.parameter.Parameter(torch.tensor(bias2, dtype=torch.float32))
    weights3 = params[hidden_dim*3+1:hidden_dim*3+2].reshape(1,1)
    weights3 = torch.nn.parameter.Parameter(torch.tensor(weights3, dtype=torch.float32))
    bias3 = params[hidden_dim*3+2:hidden_dim*3+3].reshape(1,1)
    bias3 = torch.nn.parameter.Parameter(torch.tensor(bias3, dtype=torch.float32))
    net[0].weight = weights1
    net[0].bias = bias1
    net[2].weight = weights2
    net[2].bias = bias2
    net[4].weight = weights3
    net[4].bias = bias3
    return net(torch.tensor(x, dtype=torch.float32)).detach().numpy().flatten()

def Gamma(x, params):
    ## Error model (stochastic part)
    mkl.set_num_threads(1)
    hidden_dim = 5
    net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1, 1)
        )
    weights1 = params[:hidden_dim*1].reshape(hidden_dim,1)
    weights1 = torch.nn.parameter.Parameter(torch.tensor(weights1, dtype=torch.float32))
    bias1 = params[hidden_dim*1:hidden_dim*2].reshape(1,hidden_dim)
    bias1 = torch.nn.parameter.Parameter(torch.tensor(bias1, dtype=torch.float32))
    weights2 = params[hidden_dim*2:hidden_dim*3].reshape(1,hidden_dim)
    weights2 = torch.nn.parameter.Parameter(torch.tensor(weights2, dtype=torch.float32))
    bias2 = params[hidden_dim*3:hidden_dim*3+1].reshape(1,1)
    bias2 = torch.nn.parameter.Parameter(torch.tensor(bias2, dtype=torch.float32))
    weights3 = params[hidden_dim*3+1:hidden_dim*3+2].reshape(1,1)
    weights3 = torch.nn.parameter.Parameter(torch.tensor(weights3, dtype=torch.float32))
    bias3 = params[hidden_dim*3+2:hidden_dim*3+3].reshape(1,1)
    bias3 = torch.nn.parameter.Parameter(torch.tensor(bias3, dtype=torch.float32))
    net[0].weight = weights1
    net[0].bias = bias1
    net[2].weight = weights2
    net[2].bias = bias2
    net[4].weight = weights3
    net[4].bias = bias3
    return net(torch.tensor(x, dtype=torch.float32)).detach().numpy().flatten()

def G(u, z0, y):
    ## Forward model G
    mkl.set_num_threads(1)
    l96m = L96M(h = 10./3., c = 3.)
    l96m.set_stencil()
    l96m.set_predictor(predictor)
    l96m.set_Gamma(Gamma)
    l96m.set_params(u[:18])
    l96m.set_params_Gamma(u[18:])
    T = 120 
    dt = 0.01
    t = np.arange(0.0, T, dt)
    sol_reg = sdeint.itoint(l96m.regressed_odeint, l96m.G, \
                            z0[:l96m.K], t)
    return H(sol_reg[int(20.0/dt):,:36])

def ensembleG(ui, z0, y):
    ## Parallel computing of forward model G
    Gmatrix = np.zeros([ui.shape[0],153])
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
    l96m = L96M(h = 10./3., c = 3.)
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
    y = np.loadtxt('../data/L96_c_3_y_mean.dat')
    y_cov = np.loadtxt('../data/L96_c_3_y_cov.dat')
    y_cov = np.diag(np.diag(y_cov))
    
    ## Set up EKI to calibrate the parameters in error model
    DAsteps = 30 
    numS = 200 
    params = np.ones([numS,int(36)])
    for iterN in range(params.shape[1]):
        params[:,iterN] = np.random.uniform(-1., 1, numS) 
    
    ## Run EKI
    eki = EKI(params, y, y_cov, 1)
    for iterN in range(DAsteps):
        print('DA step: ', iterN+1)
        eki.EnKI(ensembleG(eki.u[iterN], z0_conv, y))
        print("Error: ", eki.error[-1])
        pickle.dump(eki.u, open("L96_c_3_indirect_stochastic_u.pkl", "wb"))
        pickle.dump(eki.g, open("L96_c_3_indirect_stochastic_g.pkl", "wb"))
