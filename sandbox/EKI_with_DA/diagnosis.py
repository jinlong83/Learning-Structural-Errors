import os
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pdb import set_trace as bp

matplotlib.rcParams["text.usetex"] = True

# Add global settings for larger font sizes
font_size = 20
legend_font_size = 16  # Adjust this value as needed
param_dict = {
    "font.size": font_size,
    "axes.labelsize": font_size,
    "axes.titlesize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": legend_font_size,
}
matplotlib.rcParams.update(param_dict)


def plotPhi(steps, Phi, filename):
    plt.figure()
    plt.plot(steps, Phi * 100, "o-", color="b")
    plt.xticks(steps)
    plt.xlabel("EnKI Steps")
    plt.ylabel(r"$\|y-G(\theta)\|_{\Gamma}/\|y\|_{\Gamma} (\%)$")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plotTheta(steps, mean, var, truth, ylabel, filename):
    plt.figure()
    plt.plot(steps, mean, "o-", color="b", label="Ensemble mean")
    plt.fill_between(
        steps,
        mean - 2 * var,
        mean + 2 * var,
        color="gray",
        alpha=0.2,
        label=r"$2\sigma$",
    )
    if not np.isnan(truth):
        plt.hlines(truth, np.min(steps), np.max(steps), "r", "dashed", label="Truth")
    lg = plt.legend(loc=0)
    lg.draw_frame(False)
    plt.xticks(steps)
    plt.xlabel("EnKI Steps")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plotObsErr(ax, mean, err, cl, mk, lg):
    n_obs = mean.shape[0]
    ax.errorbar(
        np.arange(n_obs),
        mean,
        err,
        color=cl,
        ls="None",
        marker=mk,
        capsize=5,
        markersize=5,
        label=lg,
    )


def plotAll(truth, pkl_dir=".", param_names=None, state_names=None):
    dir_name = os.path.join(pkl_dir, "figs")
    param_dir_name = os.path.join(dir_name, "params")
    if not os.path.exists(param_dir_name):
        os.makedirs(param_dir_name)

    # truth = [10.0, 28.0, 8./3., 1., 0., 0.]
    # truth = [4, 94, 209]  # , [9, 16, 10000]]

    if param_names is None:
        param_names = [r"$\theta_{i}$" for i in range(len(truth))]
    # param_names = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$'] #, r'$x_0$', r'$y_0$', r'$z_0$']

    if state_names is None:
        state_names = [r"$x_{i}$" for i in range(len(truth))]

    fin = os.path.join(pkl_dir, "error.pkl")
    phi = pickle.load(open(fin, "rb"))
    DAsteps = phi.shape[0]
    plotPhi(np.arange(DAsteps), phi, os.path.join(dir_name, "Phi.pdf"))

    ## Plot EKI parameters
    fin = os.path.join(pkl_dir, "u.pkl")
    theta = pickle.load(open(fin, "rb"))
    mean = np.mean(theta, 1)
    var = np.sqrt(np.var(theta, 1))

    for iterN in range(len(truth)):
        plotTheta(
            np.arange(DAsteps + 1),
            mean[:, iterN],
            var[:, iterN],
            truth[iterN],
            param_names[iterN],
            os.path.join(param_dir_name, "theta" + str(iterN) + ".pdf"),
        )

    # Plot G's
    fin = os.path.join(pkl_dir, "y_mean.pkl")
    y_mean = pickle.load(open(fin, "rb"))

    fin = os.path.join(pkl_dir, "y_cov.pkl")
    y_cov = pickle.load(open(fin, "rb"))
    y_err = np.sqrt(np.diagonal(y_cov))

    fin = os.path.join(pkl_dir, "H_obs.pkl")
    Hobs = pickle.load(open(fin, "rb"))
    Nstates = np.linalg.matrix_rank(Hobs)  # int(np.sum(np.diag(Hobs)))

    # if y_mean.ndim == 1:
    #     Nstates = 1
    # else:
    #     Nstates = y_mean.shape[0]
    y_mean = y_mean.reshape(-1, Nstates)
    y_err = y_err.reshape(-1, Nstates)

    fin = os.path.join(pkl_dir, "g.pkl")
    G_samples_all = pickle.load(open(fin, "rb"))

    dir_name = os.path.join(pkl_dir, "figs")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.figure()
    fig, ax_list = plt.subplots(nrows=Nstates, ncols=1, sharex=True, squeeze=True)
    if Nstates == 1:
        ax_list = [ax_list]
    for iterS in range(Nstates):
        plotObsErr(ax_list[iterS], y_mean[:, iterS], y_err[:, iterS], "r", "o", "Truth")
        ax_list[iterS].set_ylabel(r"${}(t)$".format(state_names[iterS]))
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, "G_observed.pdf"))
    plt.close("all")

    figsize = (10, 7)
    matplotlib.rcParams.update(param_dict)
    for iterN in range(G_samples_all.shape[0]):
        # plt.figure(iterN)
        plt.figure(figsize=figsize)
        fig, ax_list = plt.subplots(nrows=Nstates, ncols=1, sharex=True, squeeze=True)
        if Nstates == 1:
            ax_list = [ax_list]
        G_samples = G_samples_all[iterN, :, :]
        G_mean = np.mean(G_samples, axis=0)
        G_cov = np.cov(G_samples.T)
        G_err = np.sqrt(np.diagonal(G_cov))

        G_mean = G_mean.reshape(-1, Nstates)
        G_err = G_err.reshape(-1, Nstates)
        for iterS in range(Nstates):
            plotObsErr(
                ax_list[iterS], y_mean[:, iterS], y_err[:, iterS], "r", "o", "Truth"
            )
            plotObsErr(
                ax_list[iterS], G_mean[:, iterS], G_err[:, iterS], "b", "^", "EKI"
            )
            print(state_names)
            ax_list[iterS].set_ylabel(r"${}(t)$".format(state_names[iterS]))

        ax_list[-1].set_xlabel(r"$t$ [min]")
        ax_list[0].legend(loc="upper right")
        # plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, left=0.2)
        plt.savefig(os.path.join(dir_name, "G_" + str(iterN) + ".pdf"))
        plt.close("all")

    plt.close("all")
    return


def get_slices(arr, n):
    result = []
    for i in range(n):
        indices = [i] + [0] * (n - i) + [slice(None)] + [0] * (n - i)
        result.append(arr[tuple(indices)])
    return result


def plot_function_comparisons(x_grid, f_true, f_nn, fig_path):
    print("Plotting function comparisons and saving to {}".format(fig_path))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    n_dims = x_grid.shape[0]
    for i in range(n_dims):
        mean_axes = tuple([j for j in range(n_dims) if j != i])
        x_grid_i = np.mean(x_grid[i], axis=mean_axes)
        plt.figure(figsize=(1, 1))
        plt.plot(x_grid_i, np.mean(f_true, axis=mean_axes), "b", label="True")
        plt.plot(x_grid_i, np.mean(f_nn, axis=mean_axes), "r", label="NN")
        plt.xlabel(r"$x_{}$".format(i))
        plt.ylabel(r"$\delta(x_{}, \ \cdot)$".format(i))
        plt.legend()
        plt.savefig(os.path.join(fig_path, "f_{}.pdf".format(i)))
        plt.close()
    return
