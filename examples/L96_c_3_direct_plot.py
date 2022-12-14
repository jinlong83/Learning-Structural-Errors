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
import scipy.integrate as scint
import sdeint
import pickle

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

np.random.seed(10)

def predictor(x, params = None):
    ## Error model
    net = torch.load('L96_c_3_direct_trained.pt')
    return net(torch.tensor(x, dtype=torch.float32)).detach().numpy().flatten()

def G(z0):
    ## Simulate a long trajectory
    l96m = L96M(h = 10./3., c = 3.)
    l96m.set_stencil()
    l96m.set_predictor(predictor)
    l96m.set_params(None)
    T = 1e3
    dt = 0.01
    t = np.arange(0.0, T, dt)
    sol_reg = scint.solve_ivp(
            l96m.regressed,
            [0,T],
            z0[:l96m.K],
            method = 'LSODA',
            t_eval = t,
            max_step = dt)
    sol_reg = sol_reg.y
    return sol_reg

def plot_measure(state):
    ## Plot the invariant measure
    fin = '../data/L96_c_3_direct_full_data.pkl'
    states_truth = pickle.load(open(fin, 'rb'))
    states_truth = states_truth.flatten()
    
    cwd = os.getcwd()
    plt.rcParams.update({'font.size': 16})
    plt.figure()
    state = state.flatten()
    plt.hist(state, bins = 'auto', density = True,
             histtype = 'step', color = 'b', label = 'Model', lw = 2)
    plt.hist(states_truth, bins = 'auto', density = True,
             histtype = 'step', label = 'Truth', color = 'r', lw = 2)
    plt.xlabel('X')
    plt.ylabel('Probability density')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(cwd + '/L96_c_3_direct_invariant_measure.pdf')
    plt.close()

if __name__ == "__main__":
    print("Number of cpu : ", multiprocessing.cpu_count())
    print('Start plotting:')
    start = time.time()
    z0 = np.loadtxt('../data/L96_c_3_x0.dat')
    state = G(z0)
    plot_measure(state)
    end = time.time()
    print('Time elapsed: ', end - start)
