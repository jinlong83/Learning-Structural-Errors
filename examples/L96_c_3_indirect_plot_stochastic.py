import sys
sys.path.append('../src')
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy as sp
import multiprocessing
from L96M_ import L96M
from enki import EKI
import scipy.integrate as scint
import sdeint
import pickle
import mkl

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

np.random.seed(10)

def z0_calc():
    ## Randomly simulate an initial condition
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
    return z0_conv[:36]

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

def G(u, z0):
    ## Simulate a long trajectory based on given parameters of the error model
    mkl.set_num_threads(1)
    l96m = L96M(h = 10./3., c = 3.)
    l96m.set_stencil()
    l96m.set_predictor(predictor)
    l96m.set_Gamma(Gamma)
    l96m.set_params(u[:18])
    l96m.set_params_Gamma(u[18:])
    T = 1e3
    dt = 0.01
    t = np.arange(0.0, T, dt)
    sol_reg = sdeint.itoint(l96m.regressed_odeint, l96m.G, \
                            z0[:l96m.K], t)
    sol_reg = sol_reg.T
    return sol_reg

def ensembleG(ui, z0):
    ## Simulate long trajectories for the ensemble of calibrated parameters
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = []
    for iterN in range(ui.shape[0]):
        results.append(pool.apply_async(G, (ui[iterN,:],z0)))
    iterN = 0
    Gmatrix = np.zeros([ui.shape[0], 36, int(1e5)])
    for result in results:
        Gmatrix[iterN,:,:] = result.get()
        iterN = iterN+1
    pool.close()
    return Gmatrix

def plot_measure(states):
    ## Plot the invariant measure
    fin = '../data/L96_c_3_direct_full_data.pkl'
    states_truth = pickle.load(open(fin, 'rb'))
    states_truth = states_truth.flatten()
    cwd = os.getcwd()
    plt.rcParams.update({'font.size': 16})
    plt.figure()
    for iterS in range(states.shape[0]):
        state = states[iterS,:,:] 
        state = state.flatten()
        plt.hist(state, bins = 'auto', density = True,
                 histtype = 'step', color = 'b', lw = 2)
    plt.hist(states_truth, bins = 'auto', density = True,
             histtype = 'step', label = 'Truth', color = 'r', lw = 2)
    plt.xlabel('X')
    plt.ylabel('Probability density')
    plt.tight_layout()
    plt.savefig(cwd + '/L96_c_3_indirect_stochastic_invariant_measure.pdf')
    plt.close()

def plot_data():
    ## Plot the truth-prediction comparison for the data
    matplotlib.rcParams.update({'font.size':16})
    y_mean = np.loadtxt('../data/L96_c_3_y_mean.dat')
    y_cov = np.loadtxt('../data/L96_c_3_y_cov.dat')
    fin = 'L96_c_3_indirect_stochastic_g.pkl'
    G = pickle.load(open(fin, 'rb'))
    G = G[-1,:,:]
    G_mean = np.mean(G, axis = 0)
    G_cov = np.cov(G.T)
    x_min = -10
    x_max = 15
    plt.plot([x_min,x_max], [x_min,x_max], 'k--', lw = 2)
    plt.plot(y_mean[:8], G_mean[:8], 'o', markersize=12, color = "#a6cee3", alpha=1.0, label='First moments')
    plt.plot(y_mean[8:], G_mean[8:], '^', markersize=12, color = "#1f78b4", alpha=1.0, label='Second moments')
    plt.xlim((x_min,x_max))
    plt.ylim((x_min,x_max))
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.legend(frameon=False)
    plt.tight_layout()
    cwd = os.getcwd()
    plt.savefig(cwd + '/L96_c_3_indirect_stochastic_data.pdf')

if __name__ == "__main__":
    print("Number of cpu : ", multiprocessing.cpu_count())
    print('Start plotting:')
    start = time.time()
    fin = 'L96_c_3_indirect_stochastic_u.pkl'
    params_all = pickle.load(open(fin, 'rb'))
    params = params_all[-1,:,:]
    z0 = z0_calc()
    states = ensembleG(params, z0)
    plot_measure(states)
    plot_data()
    end = time.time()
    print('Time elapsed: ', end - start)
