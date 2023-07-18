import os
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pdb import set_trace as bp

matplotlib.rcParams['text.usetex'] = True

def plotPhi(steps, Phi, filename):
    plt.figure()
    plt.plot(steps, Phi*100, 'o-', color='b')
    plt.xticks(steps)
    plt.xlabel('EnKI Steps')
    plt.ylabel(r"$\|y-G(\theta)\|_{\Gamma}/\|y\|_{\Gamma} (\%)$")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plotTheta(steps, mean, var, truth, ylabel, filename):
    plt.figure()
    plt.plot(steps, mean, 'o-', color='b', label='Ensemble mean')
    plt.fill_between(steps, mean -2*var, mean + 2*var,
                     color='gray', alpha=0.2, label=r'$2\sigma$')
    if not np.isnan(truth):
        plt.hlines(truth, np.min(steps), np.max(steps), 'r', 'dashed', label='Truth')
#     lg = plt.legend(loc=0)
#     lg.draw_frame(False)
    plt.xticks(steps)
    plt.xlabel('EnKI Steps')
#     plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plotObsErr(ax, mean, err, cl, mk, lg):
    n_obs = mean.shape[0]
    ax.errorbar(np.arange(n_obs), mean, err, color = cl, ls = 'None',\
                 marker = mk, capsize=5, markersize = 5, label = lg)

def plotAll(truth, pkl_dir='.', ylabels=None):

    dir_name = os.path.join(pkl_dir,'figs')
    param_dir_name = os.path.join(dir_name, 'params')
    if not os.path.exists(param_dir_name):
        os.makedirs(param_dir_name)
    

    # truth = [10.0, 28.0, 8./3., 1., 0., 0.]
    # truth = [4, 94, 209]  # , [9, 16, 10000]]

    if ylabels is None:
        ylabels = [r'$\theta_{i}$' for i in range(len(truth))]
    # ylabels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$'] #, r'$x_0$', r'$y_0$', r'$z_0$']

    fin = os.path.join(pkl_dir, 'error.pkl')
    phi = pickle.load(open(fin, 'rb'))
    DAsteps = phi.shape[0]
    plotPhi(np.arange(DAsteps), phi, os.path.join(dir_name,'Phi.pdf'))


    ## Plot EKI parameters
    fin = os.path.join(pkl_dir, 'u.pkl')
    theta = pickle.load(open(fin, 'rb'))
    mean = np.mean(theta, 1)
    var = np.sqrt(np.var(theta, 1))

    for iterN in range(len(truth)):
        plotTheta(np.arange(DAsteps+1), mean[:, iterN], var[:, iterN], truth[iterN],
                  ylabels[iterN], os.path.join(param_dir_name,'theta'+str(iterN)+'.pdf'))

    # Plot G's
    fin = os.path.join(pkl_dir,'y_mean.pkl')
    y_mean = pickle.load(open(fin, 'rb'))

    fin = os.path.join(pkl_dir,'y_cov.pkl')
    y_cov = pickle.load(open(fin, 'rb'))
    y_err = np.sqrt(np.diagonal(y_cov))

    fin = os.path.join(pkl_dir,'H_obs.pkl')
    Hobs = pickle.load(open(fin, 'rb'))
    Nstates = np.linalg.matrix_rank(Hobs) #int(np.sum(np.diag(Hobs)))

    # if y_mean.ndim == 1:
    #     Nstates = 1
    # else:
    #     Nstates = y_mean.shape[0]
    y_mean = y_mean.reshape(-1, Nstates)
    y_err = y_err.reshape(-1, Nstates)

    fin = os.path.join(pkl_dir,'g.pkl')
    G_samples_all = pickle.load(open(fin, 'rb'))

    dir_name = os.path.join(pkl_dir,'figs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.figure()
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)
    ax_list = [ax1, ax2, ax3]
    for iterS in range(Nstates):
        plotObsErr(ax_list[iterS], y_mean[:, iterS],
                y_err[:, iterS], 'r', 'o', 'Truth')
    ax1.set_ylabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax3.set_ylabel(r'$z$')
    ax3.set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name,'G_observed.pdf'))
    plt.close('all')

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

        G_mean = G_mean.reshape(-1,Nstates)
        G_err = G_err.reshape(-1,Nstates)
        for iterS in range(Nstates):
            plotObsErr(ax_list[iterS], y_mean[:,iterS], y_err[:,iterS], 'r', 'o', 'Truth')
            plotObsErr(ax_list[iterS], G_mean[:,iterS], G_err[:,iterS], 'b', '^', 'EKI')
        
        ax1.set_ylabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax3.set_ylabel(r'$z$')
        ax3.set_xlabel('EKI steps')
        ax1.legend(loc = 'upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(dir_name, 'G_' + str(iterN) + '.pdf'))

    plt.close('all')
    return