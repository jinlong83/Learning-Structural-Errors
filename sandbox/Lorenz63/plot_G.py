import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os

matplotlib.rcParams.update({'font.size':16})

def plotObsErr(ax, mean, err, cl, mk, lg):
    n_obs = mean.shape[0]
    ax.errorbar(np.arange(n_obs), mean, err, color = cl, ls = 'None',\
                 marker = mk, capsize=5, markersize = 5, label = lg)

fin = 'y_mean.pkl'
y_mean = pickle.load(open(fin, 'rb'))

fin = 'y_cov.pkl'
y_cov = pickle.load(open(fin, 'rb'))
y_err = np.sqrt(np.diagonal(y_cov))

y_mean = y_mean.reshape(-1,3)
y_err = y_err.reshape(-1,3)

fin = 'g.pkl'
G_samples_all = pickle.load(open(fin, 'rb'))

dir_name = 'figs'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

for iterN in range(G_samples_all.shape[0]):

    plt.figure(iterN)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)
    ax_list = [ax1, ax2, ax3]

    G_samples = G_samples_all[iterN,:,:]
    G_mean = np.mean(G_samples, axis = 0)
    G_cov = np.cov(G_samples.T)
    G_err = np.sqrt(np.diagonal(G_cov))

    G_mean = G_mean.reshape(-1,3)
    G_err = G_err.reshape(-1,3)
    for iterS in range(3):
        plotObsErr(ax_list[iterS], y_mean[:,iterS], y_err[:,iterS], 'r', 'o', 'Truth')
        plotObsErr(ax_list[iterS], G_mean[:,iterS], G_err[:,iterS], 'b', '^', 'EKI')
    
    ax1.set_ylabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax3.set_ylabel(r'$z$')
    ax3.set_xlabel('EKI steps')
    plt.tight_layout()
    plt.savefig(dir_name + '/G_' + str(iterN) + '.pdf')
    plt.close('all')
