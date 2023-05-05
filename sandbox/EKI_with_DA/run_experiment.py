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
        ic_true=None,
        t_range=None,
        dt=None,
        t_da=None,
        parallel_flag=True,
        normalizer=InactiveNormalizer,
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
        self.ic_true = ic_true
        self.t_range = t_range
        self.dt = dt
        self.t_da = t_da
        self.parallel_flag = parallel_flag
        self.normalizer = normalizer
        self.da_settings = da_settings
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

    def H(self, states):
        """
        Observation function
        Current observation: flattened 1-D data of all trajectories
        """
        obs = states.flatten()  
        return obs

    def G_da(self, params_all, obs_warmup, true_warmup, times_warmup, 
             times_rollout, fig_path='.'):
        # return solution in observed components
        pred_rollout = self.WRAP.G(obs_warmup, true_warmup, times_warmup, times_rollout,
                            params=params_all, param_names=self.param_names, 
                            fig_path=fig_path, make_plots=False)
        # normalize, then flatten
        return self.H(self.normalizer.encode(pred_rollout))

    def G_joint(self, params_all, times):
        n_params = len(self.param_names)
        params = params_all[:n_params]
        state0 = params_all[n_params:] # dont normalize i.c.'s because we may not observe them anyway!

        # set parameters of the ODE
        for i, name in enumerate(self.param_names):
            setattr(self.WRAP.DA.ode, name, params[i])

        # solve system
        states = self.WRAP.solve(state0, times)

        return self.H(self.normalizer.encode((self.da_settings['H']@states.T).T))

    def G(self, params_all, obs_warmup, true_warmup, times_warmup,
           times_rollout, fig_path='.'):
        if self.use_da:
            return self.G_da(params_all, obs_warmup, true_warmup, 
                             times_warmup, times_rollout, fig_path)
        else:
            times = np.hstack((times_warmup, times_rollout))
            return self.G_joint(params_all, times)

    def ensembleG(self, ui, obs_warmup, true_warmup, times_warmup,
                   times_rollout, n_measurements=600):
        Gmatrix = np.zeros([ui.shape[0],n_measurements])
        if self.parallel_flag == True:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            results = []
            for iterN in range(ui.shape[0]):
                results.append(pool.apply_async(self.G, (ui[iterN,:], 
                        obs_warmup, true_warmup, times_warmup, times_rollout)))
            iterN = 0
            for result in results:
                Gmatrix[iterN,:] = result.get()
                iterN = iterN + 1
            pool.close()
        else:
            for iterN in range(ui.shape[0]):
                Gmatrix[iterN,:] = self.G(ui[iterN,:], 
                        obs_warmup, true_warmup, times_warmup, times_rollout)
        return Gmatrix

    def run(self):

        #initialize wrapper class
        self.WRAP = WRAPPER(**self.__dict__) #holds models and Data Assimilators

        # extract true parameters from model
        true_params = [getattr(self.WRAP.DA.ode, param_name) for param_name in self.param_names]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        np.random.seed(self.seed)
        start = time()

        ## Data generation with a initial condition
        traj_times, traj, traj_obs = self.WRAP.make_data(ic=self.ic_true, 
                                        t0=self.t_range[0], t_end=self.t_range[-1], step=self.dt)
        # traj_obs is noisy data in observed components
        
        self.normalizer = self.normalizer(traj_obs)

        # split data into warmup and rollout phases (for DA + EKI)
        times_warmup, obs_warmup, true_warmup, times_rollout, obs_rollout, true_rollout = self.splitData(traj_times, traj, traj_obs)

        if self.use_da:
            # use DA to infer ICs
            y_mean = self.H(self.normalizer.encode(obs_rollout))
            self.plotData(self.normalizer.encode(obs_rollout))
        else:
            # use EKI to infer ICs
            y_mean = self.H(self.normalizer.encode(traj_obs))
            self.plotData(self.normalizer.encode(traj_obs))

        ## Clip the minimum std to 0.1
        y_cov = np.diag(np.clip((y_mean * 0.05)**2, 0.01, None))

        # PARAMS
        if self.use_da:
            params_samples = np.zeros([self.nSamples,3])
        else:
            params_samples = np.zeros([self.nSamples,6])
            # STATES
            params_samples[:,3:] = np.array([self.WRAP.DA.ode.get_inits() for _ in range(self.nSamples)])

        # PARAMS
        params_samples[:,:3] = np.array([self.WRAP.DA.ode.sample_params(param_names=self.param_names) for _ in range(self.nSamples)])

        print("Number of cpu : ", multiprocessing.cpu_count())

        ## Save the data for post-processing
        pickle.dump(y_mean, open(os.path.join(self.output_dir, "y_mean.pkl"), "wb"))
        pickle.dump(y_cov, open(os.path.join(self.output_dir, "y_cov.pkl"), "wb"))

        ## Initialize EKI object
        eki = EKI(params_samples, y_mean, y_cov, 1)
        ## Iterations of EKI steps
        for iterN in tqdm(range(self.DAsteps)):
            print('DA step: ', iterN+1)
            ## Forward simulation of ensemble members
            G_results = self.ensembleG(eki.u[iterN], obs_warmup, true_warmup,
                                        times_warmup, times_rollout, n_measurements=len(y_mean)) # nSamples x nData

            ## Feed the ensemble evaluaation of G to EKI object
            eki.EnKI(G_results)
            print("Error: ", eki.error[-1])
            ## Save the current results of EKI
            pickle.dump(eki.u, open(os.path.join(self.output_dir, "u.pkl"), "wb"))
            pickle.dump(eki.g, open(os.path.join(self.output_dir, "g.pkl"), "wb"))
            pickle.dump(eki.error, open(os.path.join(self.output_dir, "error.pkl"), "wb"))
            end = time()
            print('Time elapsed: ', end - start)

            # make plots
            if self.use_da:
                labels = self.param_names
                truth = true_params
            else:
                labels = self.param_names + self.WRAP.DA.ode.state_names
                truth = np.hstack((true_params, self.ic_true))

            plotAll(truth, self.output_dir, labels)
