import os
import numpy as np
import multiprocessing
from diagnosis import plotAll, plot_function_comparisons

# from models import Ultradian
from tqdm import tqdm
import pandas as pd
from utils import UnitGaussianNormalizer, MaxMinNormalizer, InactiveNormalizer
# from enki import EKI
from time import time
import pickle
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["text.usetex"] = False

# Add global settings for larger font sizes
font_size = 30
legend_font_size = 30  # Adjust this value as needed
param_dict = {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": legend_font_size,
    }

matplotlib.rcParams.update(param_dict)


class EXPERIMENT(object):

    def __init__(
        self,
        param_names=None,
        param_type="mech",  # mech (params in ODE) or nn (params in NN)
        ic_true=None,
        error_component_index=0,
        t_range=None,
        dt=None,
        t_da=None,
        parallel_flag=True,
        normalizer=InactiveNormalizer,
        ode_settings_approx={},
        ode_settings_true={},
        da_settings=None,
        driver=None,
        dynamics_rhs="L63",
        DAsteps=10,
        nSamples=100,
        use_da=False,
        da_alg="3dvar",
        output_dir=".",
        seed=10,
        l2_reg=0,
        lam=1.0,
        sparse_threshold=0.1,
        inflation_std=1e-3,
    ):
        self.error_component_index = error_component_index  # which component of the ode RHS has an error term (1 for L63, 0 for ultradian)
        self.param_names = param_names
        self.param_type = param_type
        self.ic_true = ic_true
        self.t_range = t_range
        self.dt = dt
        self.t_da = t_da
        self.parallel_flag = parallel_flag
        self.normalizer = normalizer
        self.da_settings = da_settings
        self.ode_settings_approx = ode_settings_approx
        self.ode_settings_true = ode_settings_true
        self.driver = driver
        self.dynamics_rhs = dynamics_rhs
        self.DAsteps = DAsteps
        self.nSamples = nSamples
        self.use_da = use_da  # whether to use DA or not
        self.da_alg = da_alg
        self.output_dir = output_dir
        self.seed = seed
        self.l2_reg = l2_reg
        self.lam = lam
        self.sparse_threshold = sparse_threshold
        self.inflation_std = inflation_std

    def splitData(self, traj_times, traj, traj_obs):
        """Split trajectory data into warmup and rollout phases"""
        # warmup phase
        times_warmup = traj_times[: self.t_da]
        obs_warmup = traj_obs[: self.t_da]
        true_warmup = traj[: self.t_da]
        # rollout phase
        times_rollout = traj_times[self.t_da :]
        obs_rollout = traj_obs[self.t_da :]
        true_rollout = traj[self.t_da :]
        return (
            times_warmup,
            obs_warmup,
            true_warmup,
            times_rollout,
            obs_rollout,
            true_rollout,
        )

    def plotData(self, data):
        """
        Plot data
        """
        plt.figure()
        plt.plot(data)
        plt.savefig(os.path.join(self.output_dir, "data.pdf"))
        plt.close()

        # plot each row of data on a separate subplot
        try:
            fig, axs = plt.subplots(data.shape[1], 1, sharex=True)
            for i in range(data.shape[1]):
                axs[i].plot(data[:, i])
            plt.savefig(os.path.join(self.output_dir, "data_rows.pdf"))
            plt.close()
        except:
            pass

    def H(self, states):
        """
        Observation function
        Current observation: flattened 1-D data of all trajectories
        """
        obs = states.flatten()
        return obs

    def plotNN(self, params_all, fig_path="."):
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        params = params_all[: self.n_params]
        self.WRAP.DA.ode.nn_params = params
        self.WRAP.DA.ode.set_nn(params)

        # save the NN parameters to a file
        np.save(os.path.join(fig_path, "params.npy"), params)

        matplotlib.rcParams.update(param_dict)
        if "Ult" in fig_path:
            figsize = (10, 7)
            # First, plot the active input component
            x = np.array([50.0, 50.0, 10000, 10, 10, 10])
            Ip_grid = np.linspace(self.WRAP.DA.ode.I_pmin, self.WRAP.DA.ode.I_pmax, 100)
            f_nn_grid = np.zeros_like(Ip_grid)
            for i in range(Ip_grid.shape[0]):
                x[0] = Ip_grid[i]
                f_nn_grid[i] = self.WRAP.DA.ode.nn_eval(x)

            truth = -Ip_grid / self.WRAP_TRUE.DA.ode.tp
            plt.figure(figsize=figsize)
            plt.plot(
                Ip_grid,
                f_nn_grid,
                label=r"NN: $\delta(I_p, 50, 10000, 10, 10, 10)$",
                color="blue",
                linewidth=2,
            )
            plt.plot(
                Ip_grid,
                truth,
                label=r"Truth: $-I_p / t_p$",
                color="red",
                linewidth=2,
            )
            plt.xlabel(r"$I_p$")
            plt.ylabel(r"$\delta(I_p, \ \cdot)$")
            plt.legend()
            plt.subplots_adjust(bottom=0.2, left=0.2)
            plt.savefig(os.path.join(fig_path, "nn_Ip.pdf"))
            plt.close()

            # Next, check slices across inactive component
            Ip = 50.0
            x = np.array([Ip, 50.0, 10000, 10, 10, 10])
            G_grid = np.linspace(self.WRAP.DA.ode.Gmin, self.WRAP.DA.ode.Gmax, 100)
            truth = np.ones_like(G_grid) * (-Ip / self.WRAP_TRUE.DA.ode.tp)
            f_nn_grid = np.zeros_like(G_grid)
            for i in range(G_grid.shape[0]):
                x[2] = G_grid[i]
                f_nn_grid[i] = self.WRAP.DA.ode.nn_eval(x)
            plt.figure(figsize=figsize)
            plt.plot(
                Ip_grid,
                f_nn_grid,
                label=r"NN: $\delta(50, 50, G, 10, 10, 10)$",
                color="blue",
                linewidth=2,
            )
            plt.plot(Ip_grid, truth, label=r"Truth: $-50 / t_p$", color="red",
                     linewidth=2)
            plt.xlabel(r"$G$")
            plt.ylabel(r"$\delta(G, \ \cdot)$")
            plt.legend()
            plt.subplots_adjust(bottom=0.2, left=0.2)
            plt.savefig(os.path.join(fig_path, "nn_G.pdf"))
            plt.close()

            x = np.array([Ip, 50.0, 10000, 10, 10, 10])
            Ii_grid = np.linspace(self.WRAP.DA.ode.I_imin, self.WRAP.DA.ode.I_imax, 100)
            f_nn_grid = np.zeros_like(Ii_grid)
            for i in range(Ii_grid.shape[0]):
                x[1] = Ii_grid[i]
                f_nn_grid[i] = self.WRAP.DA.ode.nn_eval(x)
            plt.figure(figsize=figsize)
            plt.plot(
                Ii_grid,
                f_nn_grid,
                label=r"NN: $\delta(50, I_i, 10000, 10, 10, 10)$",
                color="blue",
                linewidth=2,
            )
            plt.plot(Ii_grid, truth, label=r"Truth: $-50 / t_p$", color="red",
                     linewidth=2)
            plt.xlabel(r"$I_i$")
            plt.ylabel(r"$\delta(I_i, \ \cdot)$")
            plt.legend()
            plt.subplots_adjust(bottom=0.2, left=0.2)
            plt.savefig(os.path.join(fig_path, "nn_Ii.pdf"))
            plt.close()
        elif "63" in fig_path:
            pass
        else:
            raise ValueError("Unknown model")

        return

    def G(self, params_all, obs, true, times, fig_path="."):
        self.plotNN(params_all, fig_path)

        params = params_all[: self.n_params]
        # state0 = np.array([50., 50., 10000, 10, 10, 10]) # + np.random.randn(3)
        # print('Warning: using fixed initial conditions')

        # set parameters of the ODE
        # for i, name in enumerate(self.param_names):
        #     setattr(self.WRAP.DA.ode, name, params[i])
        self.WRAP.DA.ode.nn_params = params
        self.WRAP.DA.ode.set_nn(params)

        if self.use_da:
            # return solution in observed components
            pred_rollout = self.WRAP.G(
                obs,
                true,
                times,
                params=params_all,
                param_names=self.param_names,
                fig_path=fig_path,
                make_plots=True,
            )

            # use self.t_da to discard predictions made during transient assimilation phase
            pred_final = pred_rollout[self.t_da :]
        else:
            state0 = params_all[
                self.n_params :
            ]  # dont normalize i.c.'s because we may not observe them anyway!
            # rescale initial conditions (-1,1) to original scale
            state0 = self.WRAP.DA.ode.rescale_states(state0)
            # solve system
            states = self.WRAP.solve(state0, times)
            pred_final = (self.da_settings["H"] @ states.T).T

        # plot f_true and f_nn
        # x_grid = self.WRAP.DA.ode.x_grid

        # # feed batches in first dimensions
        # f_nn = self.WRAP.DA.ode.nn_eval(x_grid.T).T
        # #

        # # feed batches in last dimensions
        # f_true = self.WRAP_TRUE.DA.ode.rhs(x_grid, 0, add_correction=False) - self.WRAP.DA.ode.rhs(x_grid, 0, add_correction=False)
        # plot_function_comparisons(x_grid, f_true[self.error_component_index], f_nn, fig_path)

        # normalize, then flatten
        flattened_data = self.H(self.normalizer.encode(pred_final))
        if self.l2_reg > 0:
            # concatenate flattened data and NN parameters
            flattened_data = np.concatenate([flattened_data, params])

        return flattened_data

    def ensembleG(self, ui, obs, true, times, fig_path=".", n_measurements=600):
        Gmatrix = np.zeros([ui.shape[0], n_measurements])
        if self.parallel_flag == True:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            results = []
            for iterN in range(ui.shape[0]):
                iterPath = os.path.join(fig_path, "ensembleG_particle" + str(iterN))
                results.append(
                    pool.apply_async(self.G, (ui[iterN, :], obs, true, times, iterPath))
                )
            iterN = 0
            for result in results:
                Gmatrix[iterN, :] = result.get()
                iterN = iterN + 1
            pool.close()
        else:
            for iterN in range(ui.shape[0]):
                iterPath = os.path.join(fig_path, "ensembleG_particle" + str(iterN))
                Gmatrix[iterN, :] = self.G(ui[iterN, :], obs, true, times, iterPath)
        return Gmatrix

    def run(self):
        np.random.seed(self.seed)
        from enki_cvxopt import EKI  # must import AFTER setting random seed
        from DA import WRAPPER

        # initialize wrapper class
        # holds models and Data Assimilators
        self.WRAP_TRUE = WRAPPER(ode_settings=self.ode_settings_true, **self.__dict__)

        self.WRAP = WRAPPER(
            ode_settings=self.ode_settings_approx, **self.__dict__
        )  # holds models and Data Assimilators

        n_states = self.WRAP.DA.dim_x

        # extract true parameters from model
        if self.param_type == "mech":
            true_params = [
                getattr(self.WRAP_TRUE.DA.ode, param_name)
                for param_name in self.param_names
            ]
            self.n_params = len(self.param_names)
        elif self.param_type == "nn":
            self.n_params = self.WRAP.DA.ode.nn_num_params
            # define parameter names as theta indexed by i with superscript NN in latex format
            self.param_names = [
                r"$\theta^{(NN)}_{" + str(i) + "}$" for i in range(self.n_params)
            ]
            # include state names if not using DA
            if not self.use_da:
                self.param_names += self.WRAP.DA.ode.state_names

            true_params = np.nan * np.ones(
                len(self.param_names)
            )  # store truth as zeros

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        start = time()

        ## Data generation with a initial condition
        traj_times, traj, traj_obs = self.WRAP_TRUE.make_data(
            ic=self.ic_true, t0=self.t_range[0], t_end=self.t_range[-1], step=self.dt
        )
        # traj_obs is noisy data in observed components

        self.normalizer = self.normalizer(traj_obs)

        y_mean = self.H(self.normalizer.encode(traj_obs[self.t_da :]))
        if self.l2_reg > 0:
            y_mean = np.concatenate([y_mean, np.zeros(self.n_params)])
        self.plotData(self.normalizer.encode(traj_obs))

        # y_cov = np.diag(np.clip((y_mean * 0.05)**2, 0.01, None)) ## Clip the minimum std to 0.1
        # y_cov = (10000*self.WRAP.DA.obs_noise_sd)**2 * np.eye(len(y_mean))
        diag_weights = np.ones(len(y_mean))
        if self.l2_reg > 0:
            # create a vector [1,1,1,...l2_reg, l2_reg, l2_reg]
            diag_weights[-self.n_params :] = self.l2_reg**2

        y_cov = np.diag(diag_weights)

        print("WARNING: assuming observation noise covariance = I")

        # PARAMS
        # params_samples = np.zeros([self.nSamples,self.n_params])
        # print('Warning: using fixed initial conditions and not inferring them w/ EKI')
        if self.use_da:
            params_samples = np.zeros([self.nSamples, self.n_params])
        else:
            params_samples = np.zeros([self.nSamples, self.n_params + n_states])
            # STATES
            # params_samples[:, self.n_params:] = np.array(
            #     [self.WRAP.DA.ode.get_inits() for _ in range(self.nSamples)])

        # PARAMS drawn from uniform on -1 to 1
        if self.param_type == "nn":
            params_samples = np.random.uniform(
                -1, 1, size=(self.nSamples, self.n_params + n_states)
            )
            # params_samples[:, :self.n_params] = np.random.uniform(-1, 1, size=(self.nSamples, self.n_params))
            # params_samples[:, 1:] *= 0.001
            # params_samples[:, 0] = -1/6 #0
            # params_samples[:, 1] = 0 #-0.01
            # params_samples[:, 2] = 0
            # params_samples[:, 3] = 0

            if self.WRAP.DA.ode.nn_rescale_input:
                n_in = 2 * self.WRAP_TRUE.DA.ode.nn_dims[0]
                # first parameters the per-state bias and sd for nn inputs
                params_samples[:, :n_in] *= np.random.uniform(
                    -5, 0, size=(self.nSamples, n_in)
                )  # look across 10^-4 to 10^4
            if self.WRAP.DA.ode.nn_rescale_output:
                n_out = 2 * self.WRAP_TRUE.DA.ode.nn_dims[-1]
                # last parameters are the per-state bias and sd for nn outputs
                # look across 10^-4 to 10^4
                params_samples[:, -n_out:] *= np.random.uniform(
                    -2, 2, size=(self.nSamples, n_out)
                )
        elif self.param_type == "mech":
            params_samples[:, : self.n_params] = np.array(
                [
                    self.WRAP.DA.ode.sample_params(param_names=self.param_names)
                    for _ in range(self.nSamples)
                ]
            )
            # note: missing a good initialization for the ICs! fix this if doing pure mech. inference

        print("Number of cpu : ", multiprocessing.cpu_count())

        ## Save the data for post-processing
        pickle.dump(y_mean, open(os.path.join(self.output_dir, "y_mean.pkl"), "wb"))
        pickle.dump(y_cov, open(os.path.join(self.output_dir, "y_cov.pkl"), "wb"))
        Hobs = self.WRAP.DA.H
        pickle.dump(Hobs, open(os.path.join(self.output_dir, "H_obs.pkl"), "wb"))

        ## Try to run plots of existing run
        # load u
        try:
            u = pickle.load(open(os.path.join(self.output_dir, "u.pkl"), "rb"))
            n_iters = u.shape[0] - 1
            print("Plotting existing runs")
            # plot final iteration
            iterN = n_iters
            fig_path = os.path.join(self.output_dir, "figs/EKI_iter" + str(iterN))
            G_results = self.ensembleG(
                u[iterN],
                traj_obs,
                traj,
                traj_times,
                fig_path,
                n_measurements=len(y_mean),
            )
            # for iterN in reversed(range(n_iters)):
            #     print('Plotting iteration ', iterN)
            #     fig_path = os.path.join(self.output_dir, 'figs/EKI_iter'+str(iterN))
            #     G_results = self.ensembleG(u[iterN], traj_obs, traj,
            #                             traj_times, fig_path, n_measurements=len(y_mean))
            return
        except:
            print("No existing run to plot")

        ## Initialize EKI object
        eki = EKI(params_samples, y_mean, y_cov, 1,
                  lam=self.lam,
                  sparse_threshold=self.sparse_threshold,
                  inflation_std=self.inflation_std)
        ## Iterations of EKI steps
        for iterN in tqdm(range(self.DAsteps)):
            print("DA step: ", iterN + 1)
            fig_path = os.path.join(self.output_dir, "figs/EKI_iter" + str(iterN))
            ## Forward simulation of ensemble members
            G_results = self.ensembleG(
                eki.u[iterN],
                traj_obs,
                traj,
                traj_times,
                fig_path,
                n_measurements=len(y_mean),
            )  # nSamples x nData

            ## Feed the ensemble evaluaation of G to EKI object
            # if last 2 iterations, omit inflation
            inflate_u = iterN < self.DAsteps - 5
            eki.EnKI(G_results, inflate_u=inflate_u)

            print("Error: ", eki.error[-1])
            ## Save the current results of EKI
            pickle.dump(eki.u, open(os.path.join(self.output_dir, "u.pkl"), "wb"))
            pickle.dump(eki.g, open(os.path.join(self.output_dir, "g.pkl"), "wb"))
            pickle.dump(
                eki.error, open(os.path.join(self.output_dir, "error.pkl"), "wb")
            )
            end = time()
            print("Time elapsed: ", (end - start) / 60, " minutes")

            # make plots
            if self.use_da:
                labels = self.param_names
                truth = true_params
            else:
                labels = self.param_names + self.WRAP.DA.ode.state_names
                truth = np.hstack((true_params, self.ic_true))

            # tem to never do state IC inference
            labels = self.param_names
            truth = true_params

            state_names = [
                self.WRAP.DA.ode.state_names[int(ind)]
                for ind in np.where(self.WRAP.DA.H)[-1]
            ]
            print(state_names)
            plotAll(
                truth,
                self.output_dir,
                param_names=self.param_names,
                state_names=state_names,
            )
