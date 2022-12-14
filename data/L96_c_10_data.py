import sys
sys.path.append('../src')
import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
import multiprocessing
import pickle

from L96M_ import L96M
import scipy.integrate as scint

np.random.seed(10)

T = 1e3 
dt = 0.01
l96m = L96M()
l96m.set_stencil()
z0 = np.empty(l96m.K + l96m.K * l96m.J)
T_conv = 50
dt_conv = dt 
t_range = np.arange(0, T, 0.01)

## Simulate an initial condition
z0[:l96m.K] = np.random.rand(l96m.K) * 15 - 5
for k_ in range(0,l96m.K):
  z0[l96m.K + k_*l96m.J : l96m.K + (k_+1)*l96m.J] = z0[k_]
sol_conv = scint.solve_ivp(
        l96m.full,
        [0,T_conv],
        z0,
        method = 'LSODA',
        max_step = dt)
z0_conv = sol_conv.y[:,-1]
np.savetxt('L96_c_10_x0.dat', z0_conv)
del sol_conv

## Simulate a long trajectory as training data
start = time.time()
truth0 = scint.solve_ivp(
        l96m.full,
        [0,T],
        z0_conv,
        method = 'LSODA',
        t_eval = t_range,
        max_step = dt)
end = time.time()
print("Total time elapsed: ", end-start)
pickle.dump(truth0.y[:36,:], open("L96_c_10_direct_full_data.pkl", "wb" ))

## Save the closure data for direct training
pairs = l96m.gather_pairs(truth0.y)
pairs = pairs[::10, :]
np.savetxt('L96_c_10_direct.dat', pairs)
