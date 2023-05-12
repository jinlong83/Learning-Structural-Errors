import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from diagnosis import *
import pickle
import os

matplotlib.rcParams.update({'font.size':16})

DAsteps = 10

fin = 'error.pkl'
phi = pickle.load(open(fin, 'rb'))
plotPhi(np.arange(DAsteps), phi)

dir_name = 'figs'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

## Plot EKI parameters
fin = 'u.pkl'
theta = pickle.load(open(fin, 'rb'))
mean = np.mean(theta, 1)
var = np.sqrt(np.var(theta, 1))
truth = [1., 0., 0.]

ylabels = [r'$x_0$', r'$y_0$', r'$z_0$']

for iterN in range(3):
    plotTheta(np.arange(DAsteps+1), mean[:,iterN], var[:,iterN], truth[iterN], \
              ylabels[iterN], 'figs/state'+str(iterN)+'.pdf')
