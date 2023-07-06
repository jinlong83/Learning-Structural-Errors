import os
import numpy as np
import multiprocessing
from diagnosis import plotAll
# from models import Ultradian 
from DA import WRAPPER
from tqdm import tqdm
import pandas as pd
from utils import UnitGaussianNormalizer, MaxMinNormalizer, InactiveNormalizer
from enki import EKI
from time import time
import pickle
from pdb import set_trace as bp

class EXPERIMENT(object):
    def __init__(self,
        param_names=None,
        param_type='mech', # mech (params in ODE) or nn (params in NN)
        ic_true=None,
        t_range=None,
        dt=None,
        t_da=None,
        parallel_flag=True,
        normalizer=InactiveNormalizer,
        ode_settings_approx={},
        ode_settings_true={},
        da_settings=None,
        driver=None,
        dynamics_rhs='L63',
        DAsteps=10,
        nSamples=100,
        use_da=False,
        da_alg='3dvar',
        output_dir='.',
        seed=10
        ):

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
        self.use_da = use_da # whether to use DA or not
        self.da_alg = da_alg
        self.output_dir = output_dir
        self.seed = seed

    def splitData(self, traj_times, traj, traj_obs):
        '''Split trajectory data into warmup and rollout phases'''
        # warmup phase
        times_warmup = traj_times[:self.t_da]
        obs_warmup = traj_obs[:self.t_da]
        true_warmup = traj[:self.t_da]
        # rollout phase
        times_rollout = traj_times[self.t_da:]
        obs_rollout = traj_obs[self.t_da:]
        true_rollout = traj[self.t_da:]
        return times_warmup, obs_warmup, true_warmup, times_rollout, obs_rollout, true_rollout

    def plotData(self, data):
        """
        Plot data
        """
        import matplotlib.pyplot as plt
        plt.plot(data)
        plt.savefig(os.path.join(self.output_dir, "data.pdf"))
        plt.close()

        # plot each row of data on a separate subplot
        try:
            fig, axs = plt.subplots(data.shape[1], 1, sharex=True)
            for i in range(data.shape[1]):
                axs[i].plot(data[:,i])
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

    def G(self, params_all, obs, true, times, fig_path='.'):
        params = params_all[:self.n_params]
        state0 = params_all[self.n_params:] # dont normalize i.c.'s because we may not observe them anyway!

        # set parameters of the ODE
        # for i, name in enumerate(self.param_names):
        #     setattr(self.WRAP.DA.ode, name, params[i])
        self.WRAP.DA.ode.nn_params = params
        self.WRAP.DA.ode.set_nn(params)

        if self.use_da:
            # return solution in observed components
            pred_rollout = self.WRAP.G(obs, true, times,
                                params=params_all, param_names=self.param_names, 
                                fig_path=fig_path, make_plots=True)

            # use self.t_da to discard predictions made during transient assimilation phase
            pred_final = pred_rollout[self.t_da:]
        else:
            # solve system
            states = self.WRAP.solve(state0, times)
            pred_final = (self.da_settings['H']@states.T).T

        # normalize, then flatten
        return self.H(self.normalizer.encode(pred_final))


    def ensembleG(self, ui, obs, true, times, fig_path='.', n_measurements=600):
        Gmatrix = np.zeros([ui.shape[0],n_measurements])
        if self.parallel_flag == True:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            results = []
            for iterN in range(ui.shape[0]):
                iterPath = os.path.join(fig_path, 'ensembleG_particle'+str(iterN))
                results.append(pool.apply_async(
                    self.G, (ui[iterN, :], obs, true, times, iterPath)))
            iterN = 0
            for result in results:
                Gmatrix[iterN,:] = result.get()
                iterN = iterN + 1
            pool.close()
        else:
            for iterN in range(ui.shape[0]):
                iterPath = os.path.join(fig_path, 'ensembleG_particle'+str(iterN))
                Gmatrix[iterN,:] = self.G(ui[iterN,:], obs, true, times, iterPath)
        return Gmatrix

    def run(self):

        #initialize wrapper class
        # holds models and Data Assimilators
        self.WRAP_TRUE = WRAPPER(ode_settings=self.ode_settings_true, **self.__dict__)

        self.WRAP = WRAPPER(ode_settings=self.ode_settings_approx, **self.__dict__) #holds models and Data Assimilators

        n_states = self.WRAP.DA.dim_x

        # extract true parameters from model
        if self.param_type == 'mech':
            true_params = [getattr(self.WRAP_TRUE.DA.ode, param_name)
                           for param_name in self.param_names]
            self.n_params = len(self.param_names)
        elif self.param_type == 'nn':
            self.n_params = self.WRAP.DA.ode.nn_num_params
            # define parameter names as theta indexed by i with superscript NN in latex format
            self.param_names = [r'$\theta^{(NN)}_{' + str(i) + '}$' for i in range(self.n_params)]
            true_params = np.nan*np.ones(self.n_params) # store truth as zeros
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        np.random.seed(self.seed)
        start = time()

        ## Data generation with a initial condition
        traj_times, traj, traj_obs = self.WRAP_TRUE.make_data(ic=self.ic_true, 
                                        t0=self.t_range[0], t_end=self.t_range[-1], step=self.dt)
        # traj_obs is noisy data in observed components
        
        self.normalizer = self.normalizer(traj_obs)

        y_mean = self.H(self.normalizer.encode(traj_obs[self.t_da:]))
        self.plotData(self.normalizer.encode(traj_obs))

        # y_cov = np.diag(np.clip((y_mean * 0.05)**2, 0.01, None)) ## Clip the minimum std to 0.1
        y_cov = self.WRAP.DA.obs_noise_sd**2 * np.eye(len(y_mean))

        # PARAMS
        if self.use_da:
            params_samples = np.zeros([self.nSamples,self.n_params])
        else:
            params_samples = np.zeros([self.nSamples,self.n_params+n_states])
            # STATES
            params_samples[:, self.n_params:] = np.array(
                [self.WRAP.DA.ode.get_inits() for _ in range(self.nSamples)])

        # PARAMS drawn from uniform on -1 to 1
        if self.param_type=='nn':
            params_samples[:, :self.n_params] = np.random.uniform(-1, 1, size=(self.nSamples, self.n_params))
            # params_samples[:, 1:] *= 0.001
            # params_samples[:, 0] = -1/6 #0
            # params_samples[:, 1] = 0 #-0.01
            # params_samples[:, 2] = 0
            # params_samples[:, 3] = 0
            
            if self.WRAP.DA.ode.nn_rescale_input:
                n_in = 2*self.WRAP_TRUE.DA.ode.nn_dims[0]
                # first parameters the per-state bias and sd for nn inputs
                params_samples[:, :n_in] *= np.random.uniform(-5, 0, size=(self.nSamples, n_in)) # look across 10^-4 to 10^4
            if self.WRAP.DA.ode.nn_rescale_output:
                n_out = 2*self.WRAP_TRUE.DA.ode.nn_dims[-1]
                # last parameters are the per-state bias and sd for nn outputs
                # look across 10^-4 to 10^4
                params_samples[:, -n_out:] *= np.random.uniform(-2, 2, size=(
                    self.nSamples, n_out))
        elif self.param_type=='mech':
            params_samples[:, :self.n_params] = np.array([self.WRAP.DA.ode.sample_params(
                param_names=self.param_names) for _ in range(self.nSamples)])

        print("Number of cpu : ", multiprocessing.cpu_count())

        ## Save the data for post-processing
        pickle.dump(y_mean, open(os.path.join(self.output_dir, "y_mean.pkl"), "wb"))
        pickle.dump(y_cov, open(os.path.join(self.output_dir, "y_cov.pkl"), "wb"))
        Hobs = self.WRAP.DA.H
        pickle.dump(Hobs, open(os.path.join(self.output_dir, "H_obs.pkl"), "wb"))

        ## Initialize EKI object
        eki = EKI(params_samples, y_mean, y_cov, 1)
        ## Iterations of EKI steps
        for iterN in tqdm(range(self.DAsteps)):
            print('DA step: ', iterN+1)
            fig_path = os.path.join(self.output_dir, 'figs/EKI_iter'+str(iterN))
            ## Forward simulation of ensemble members
            G_results = self.ensembleG(eki.u[iterN], traj_obs, traj,
                                        traj_times, fig_path, n_measurements=len(y_mean)) # nSamples x nData

            ## Feed the ensemble evaluaation of G to EKI object
            eki.EnKI(G_results)
            print("Error: ", eki.error[-1])
            ## Save the current results of EKI
            pickle.dump(eki.u, open(os.path.join(self.output_dir, "u.pkl"), "wb"))
            pickle.dump(eki.g, open(os.path.join(self.output_dir, "g.pkl"), "wb"))
            pickle.dump(eki.error, open(os.path.join(self.output_dir, "error.pkl"), "wb"))
            end = time()
            print('Time elapsed: ', (end - start)/60, ' minutes')

            # make plots
            if self.use_da:
                labels = self.param_names
                truth = true_params
            else:
                labels = self.param_names + self.WRAP.DA.ode.state_names
                truth = np.hstack((true_params, self.ic_true))

            plotAll(truth, self.output_dir, labels)
