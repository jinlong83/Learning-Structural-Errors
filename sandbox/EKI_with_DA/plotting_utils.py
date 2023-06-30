#!/usr/bin/env python
import os
import numpy as np
from scipy.linalg import svdvals, eigvals
from scipy.sparse.linalg import svds as sparse_svds
from scipy.sparse.linalg import eigs as sparse_eigs
import itertools

# Plotting parameters
import matplotlib
import pandas as pd
matplotlib.use("Agg")

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
import pickle
from matplotlib import colors
from matplotlib.colors import Normalize
import six
from scipy.interpolate import interpn
color_dict = dict(six.iteritems(colors.cnames))

from pdb import set_trace as bp
# font = {'size': 16}
# matplotlib.rc('font', **font)

# sns.set(rc={'text.usetex': True}, font_scale=4)


def plot_ultradian_functions(fig_path):
    from odelibrary import ULTRADIAN
    ode = ULTRADIAN()

    G = np.linspace(ode.Gmin, ode.Gmax, 100)
    Ii = np.linspace(ode.Iimin, 20*ode.Iimax, 100)
    h3 = np.linspace(ode.h3min, ode.h3max, 100)

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # F1 = ode.F1(G)
    # F2 = ode.F2(G)
    # F3 = ode.F3(Ii)
    # F4 = ode.F4(h3)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,14))

    # plot F1
    ax = axes[0,0]
    ax.plot(G, ode.F1(G), linewidth=4)
    ax.set_title(r'$F_1(G)$--Insulin production')
    ax.set_xlabel(r'$G$')
    ax.set_ylabel(r'$F_1$')

    # plot F2
    ax = axes[1,0]
    ax.plot(G, ode.F2(G), linewidth=4)
    ax.set_title(r'$F_2(G)$--Insulin-independent glucose utilization')
    ax.set_xlabel(r'$G$')
    ax.set_ylabel(r'$F_2$')

    # plot F3
    ax = axes[0,1]
    ax.plot(Ii, ode.F3(Ii), linewidth=4)
    ax.set_title(r'$F_3(I_i)$--Insulin-dependent glucose utilization')
    ax.set_xlabel(r'$I_i$')
    ax.set_ylabel(r'$F_3$')

    # plot F4
    ax = axes[1,1]
    ax.plot(h3, ode.F4(h3), linewidth=4)
    ax.set_title(r'$F_4(h_3)$--Delayed insulin-dependent glucose utilization')
    ax.set_xlabel(r'$h_3$')
    ax.set_ylabel(r'$F_4$')

    plt.savefig(fig_path)
    plt.close()

def plot_G(fig_path, H,
           data_true, times_true,
           data_obs, times_obs,
           data_assim, times_assim,
           pred_rollout, times_rollout,
           burnin=0, names=None):

    os.makedirs(fig_path, exist_ok=True)

    n_vars = data_true.shape[1]
    fig, axes = plt.subplots(nrows=n_vars, ncols=1, figsize=(12,10), sharex=True)
    obs_ind_list = np.where(H==1)[1]
    j_obs = -1

    for j in range(n_vars):
        ax = axes[j]
        ax.plot(times_true[burnin:], data_true[burnin:,j], label='True', color='gray', linewidth=4)
        if j in obs_ind_list:
            j_obs += 1
            ax.scatter(times_obs[burnin:], data_obs[burnin:,j_obs], marker='x', s=50, label='Observed', color='red')

            ax.plot(times_rollout, pred_rollout[:,j], label='Prediction', color='blue', linewidth=1)

        ax.plot(times_assim[burnin:], data_assim[burnin:,j], label='Assimilation', color='cyan', linewidth=1)
        if names is not None:
            ax.set_ylabel(r'${}$'.format(names[j]))
        ax.legend(loc='upper left')
    ax.set_xlabel('Time')
    axes[0].set_title('Trajectory assimilation and forecasts')
    plt.savefig(fig_path + '.pdf', format='pdf')
    plt.close()
    return


def plot_assimilation_traj(times, obs, true, assim, pred, fig_path, H, burnin=100, names=None):
    # plt.rcParams.update({'font.size': 64, 'legend.fontsize': 48,
    #                 'legend.facecolor': 'white', 'legend.framealpha': 0.8,
    #                 'legend.loc': 'upper left', 'lines.linewidth': 4.0})

    n_vars = true.shape[1]
    fig, axes = plt.subplots(nrows=n_vars, ncols=1, figsize=(12,10), sharex=True)
    obs_ind_list = np.where(H==1)[1]
    j_obs = -1

    for j in range(n_vars):
        ax = axes[j]
        ax.plot(times[burnin:], true[burnin:,j], label='True', color='gray', linewidth=4)
        if j in obs_ind_list:
            j_obs += 1
            ax.scatter(times[burnin:], obs[burnin:,j_obs], marker='x', s=50, label='Observed', color='red')
        if pred is not None:
            ax.scatter(times[burnin:], pred[burnin:,j], marker='o', s=50, label='Predicted', color='blue')

        ax.plot(times[burnin:], assim[burnin:,j], label='Filtered', color='cyan', linewidth=1)
        if names is not None:
            ax.set_ylabel(r'${}$'.format(names[j]))
        ax.legend(loc='right')
    ax.set_xlabel('time')
    axes[0].set_title('Trajectory assimilation')
    plt.savefig(fig_path + '.pdf', format='pdf')
    plt.close()
    return

def plot_assimilation_residual_statistics(res, fig_path):
    # plot sequence
    n_vars = res.shape[1]
    fig, ax = plt.subplots(nrows=n_vars, ncols=n_vars, figsize=(12,6))
    for i in range(n_vars): # row
        for j in range(n_vars): # col
            axfoo = ax[i,j]
            if i > j:
                density_scatter(res[:,i], res[:,j], ax = axfoo, s=5)
            elif i==j:
                sns.kdeplot(res[:,i], ax=axfoo, linewidth=4)
            else:
                axfoo.axis('off')
                continue

            if i==(n_vars-1):
                axfoo.set_xlabel('X_{}'.format(j))
            if j==0:
                axfoo.set_ylabel('X_{}'.format(i))

    fig.suptitle('Bivariate statistics')
    plt.savefig(fig_path + '.pdf', format='pdf')
    for i in range(n_vars): # row
        for j in range(n_vars): # col
            ax[i,j].set_yscale('symlog')
    plt.savefig(fig_path + '_ylog' + '.pdf', format='pdf')
    plt.close()

def plot_loss(times, loss, fig_path):
    # plot sequence
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax.plot(times, loss, linestyle='-', linewidth=4)
    ax.set_title('Loss sequence')
    ax.set_ylabel('Loss')
    ax.set_xlabel('time')
    plt.savefig(fig_path + '.pdf', format='pdf')
    ax.set_yscale('log')
    plt.savefig(fig_path + '_ylog' + '.pdf', format='pdf')
    plt.close()

def plot_K_learning(times, K_vec, fig_path):
    # plot sequence
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    for i in range(K_vec.shape[1]):
        for j in range(K_vec.shape[2]):
            ax.plot(times, K_vec[:,i,j], linestyle='-', linewidth=4, label='K_{i}{j}'.format(i=i,j=j))
    ax.set_title('K learning')
    ax.set_ylabel('K')
    ax.set_xlabel('time')
    ax.legend()
    plt.savefig(fig_path + '.pdf', format='pdf')
    plt.close()

def plot_assimilation_errors(times, errors, eps, fig_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    eps_vec = [eps for _ in range(len(times))]

    ax.plot(times, errors, linestyle='-', linewidth=4, color='blue', label='Mean Squared Error')
    ax.plot(times, eps_vec, linestyle='--', linewidth=4, color='black', label = r'$\epsilon$')
    ax.set_title('State assimilation error convergence')
    ax.set_yscale('log')
    ax.set_ylabel('MSE')
    ax.set_xlabel('time')
    plt.savefig(fig_path + '.pdf', format='pdf')
    plt.close()
