import numpy as np
import scipy.integrate as scint
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

class Lorenz63model:

    def f(self, t, state):
        theta = self.theta
        x, y, z = state  # unpack the state vector
        dX = theta[0] * (y-x) 
        dY = x * (theta[1]-z) - y 
        dZ = x*y - theta[2]*z 
        return np.array([dX, dY, dZ])

    def solve(self, params, state0, T, t_range, dt):
        self.theta = params 
        sol = scint.solve_ivp(
                self.f,
                [0,T],
                state0,
                method = 'LSODA',
                t_eval = t_range,
                max_step = dt)
        states = sol.y.transpose()
        return states

class Lorenz63modelNN:

    def create_nn(self, nn_dims):
        modules = []
        self.nn_num_layers = len(nn_dims) - 1
        for idx in range(self.nn_num_layers):
            modules.append(torch.nn.Linear(nn_dims[idx], nn_dims[idx+1]))
            if idx < self.nn_num_layers - 1:
                modules.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*modules)

    def f(self, t, state):
        theta = np.array([10.,28.,8./3.]) 
        x, y, z = state  # unpack the state vector
        dX = theta[0] * (y-x) 
        dY = x * (theta[1]-z) - self.nn(state) 
        dZ = x*y - theta[2]*z 
        return np.array([dX, dY, dZ], dtype="float64")

    def nn(self, x):
        ## Error model
        params = self.nn_params
        params_start_idx = 0
        params_end_idx = 0
        for idx in range(self.nn_num_layers):
            layer_idx = int(idx*2) 
            ## Set i-th layer weights
            weight = self.net[layer_idx].weight
            params_num = weight.flatten().shape[0]
            params_end_idx += params_num
            self.net[layer_idx].weight = torch.nn.parameter.Parameter(torch.tensor(
                                   params[params_start_idx:params_end_idx].reshape(weight.shape), 
                                   dtype=torch.float32)) 
            params_start_idx = params_end_idx
            ## Set i-th layer bias 
            bias = self.net[layer_idx].bias 
            params_num = bias.flatten().shape[0]
            params_end_idx += params_num
            self.net[layer_idx].bias = torch.nn.parameter.Parameter(torch.tensor(
                                 params[params_start_idx:params_end_idx].reshape(bias.shape),
                                 dtype=torch.float32))
            params_start_idx = params_end_idx
        return self.net(torch.tensor(x, dtype=torch.float32)).detach().numpy().flatten()

    def solve(self, params, state0, T, t_range, dt):
        self.nn_params = params 
        sol = scint.solve_ivp(
                self.f,
                [0,T],
                state0,
                method = 'LSODA',
                t_eval = t_range,
                max_step = dt)
        states = sol.y.transpose()
        return states
