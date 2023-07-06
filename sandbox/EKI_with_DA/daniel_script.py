import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from run_experiment_nn import EXPERIMENT
from DA import WRAPPER
from pdb import set_trace as bp


class FOO(object):
    def __init__(self):
        
        ## BEGIN SETTINGS
        DRIVER = np.array(pd.read_csv('../../data/P1_nutrition_expert.csv'))
        PARAM_NAMES = [] # list of parameter names to be learned
        DT = 20.0 # data timestep (not an integration timestep)
        T_RANGE = np.arange(3500, 6000, DT) # np.arange(3500, 10000, DT) # generate data every 20 minutes
        T_DA = int(0.4*len(T_RANGE)) # number of measurements to use in assimilation warmup phase
        HOBS = np.array([[0, 0, 1]])
        IC_TRUE = np.array([50., 50., 100.00]) # + np.random.randn(3)
        x_ic_mean = np.array([55., 130., 130.00])

        state_noise_cov = np.diag((0.01*(x_ic_mean)) ** 2)
        # ODE_SETTINGS_APPROX = {'Vg': 0.1, 'tp': np.inf, 'nn_dims': [3, 5, 1]}
        ODE_SETTINGS_APPROX = {'Vg': 0.1, 'nn_dims': [3, 5, 1]}
        ODE_SETTINGS_TRUE = {'Vg': 0.1}  # Vg=0.1 sets better scaling
        DA_SETTINGS = {
            't0': T_RANGE[0],
            'H': HOBS,
            'dt': DT,
            'eta': 1.0, # scale factor for 3DVAR gain matrix K = H.T / eta
            'add_state_noise': True, # add noise to generated dataset.
            'Sigma': state_noise_cov,
            'obs_noise_sd': 0.0001, #0.01,#20, # amount of noise to add to generated dataset
            'N_particles': 30, # only active if using enkf algorithm
            'integrator': 'RK45'
        }
        ## END SETTINGS

        ## Experiment 1a: joint EKI ##
        experiment = EXPERIMENT(
            parallel_flag=True, # use to parallelize G evaluation
            seed=0,
            param_type='nn',
            use_da=False,
            dynamics_rhs='UltradianGlucoseModel',
            param_names=PARAM_NAMES,
            ic_true=IC_TRUE,
            t_range=T_RANGE,
            dt=DT,
            t_da=0, # for JOINT, no need to discard transients since we infer the I.C. directly.
            driver=DRIVER,
            ode_settings_approx=ODE_SETTINGS_APPROX,
            ode_settings_true=ODE_SETTINGS_TRUE,
            da_settings=DA_SETTINGS)


        #initialize wrapper class
        # holds models and Data Assimilators
        experiment.WRAP_TRUE = WRAPPER(ode_settings=experiment.ode_settings_true, **experiment.__dict__)
        experiment.WRAP = WRAPPER(ode_settings=experiment.ode_settings_approx, **experiment.__dict__) #holds models and Data Assimilators

        # set param names and numbers
        experiment.n_params = experiment.WRAP.DA.ode.nn_num_params
        experiment.param_names = [r'$\theta^{(NN)}_{' + str(i) + '}$' for i in range(experiment.n_params)]

        ## Data generation with a initial condition
        traj_times, traj, traj_obs = experiment.WRAP_TRUE.make_data(ic=experiment.ic_true, 
                                        t0=experiment.t_range[0], t_end=experiment.t_range[-1], step=experiment.dt)
        # traj_obs is noisy data in observed components

        # optionally normalize data
        experiment.normalizer = experiment.normalizer(traj_obs)

        y_mean = experiment.H(experiment.normalizer.encode(traj_obs[experiment.t_da:]))
        experiment.plotData(experiment.normalizer.encode(traj_obs))
        y_cov = experiment.WRAP.DA.obs_noise_sd**2 * np.eye(len(y_mean))

        ## assign self stuff
        self.experiment = experiment
        self.traj_times = traj_times
        self.traj = traj
        self.traj_obs = traj_obs
        self.y_mean = y_mean
        self.y_cov = y_cov

    def make_data(self):
        '''Generate data from the true model'''
        return self.y_mean, self.y_cov

    def ensembleG(self, params_samples, parallel_flag=True):
        '''Forward simulation of ensemble members. Uses multiprocessing if parallel_flag=True'''
        self.experiment.parallel_flag = parallel_flag
        G_results = self.experiment.ensembleG(
            params_samples, self.traj_obs, self.traj, self.traj_times, n_measurements=len(self.y_mean)) # nSamples x nData
        return G_results

    def singleG(self, params_samples):
        '''Forward simulation of a single ensemble member'''
        G_results = self.experiment.G(params_samples, self.traj_obs, self.traj, self.traj_times)
        return G_results

    def computeError(self, G_results):
        '''Compute error between G_results and y_mean weighted by y_cov'''
        if G_results.ndim == 2:
            G_results_mean = np.mean(G_results, axis=0)
        else:
            G_results_mean = G_results

        diff = self.y_mean - G_results_mean
        error = diff.dot(np.linalg.solve(self.y_cov, diff))

        # normalize error
        norm = self.y_mean.dot(np.linalg.solve(self.y_cov, self.y_mean))
        error /= norm

        return error
    
if __name__ == '__main__':
    ## choose number of ensemble members
    nSamples = 30  

    ## set up data and experiment
    foo = FOO()

    ## read out the data
    y_mean, y_cov = foo.make_data()

    ## START setting the "theta" for G ##
    n_states = foo.experiment.WRAP.DA.dim_x
    n_params = foo.experiment.n_params

    if foo.experiment.use_da:
        # PARAMS
        params_samples = np.zeros([nSamples, n_params])
    else:
        # PARAMS
        params_samples = np.zeros([nSamples, n_params+n_states])

        # STATES
        params_samples[:, n_params:] = np.array(
            [foo.experiment.WRAP.DA.ode.get_inits() for _ in range(nSamples)])

    # PARAMS drawn from uniform on -1 to 1
    params_samples[:, :n_params] = np.random.uniform(-1, 1, size=(nSamples, n_params))
    ## END setting the "theta" for G ##

    ## run the forward model once
    G_results_single = foo.singleG(params_samples[0])
    print('Normalized Covariance-weighted Error: ', foo.computeError(G_results_single))

    ## run the forward model in parallel over ensemble members
    G_results_ens = foo.ensembleG(params_samples, parallel_flag=True)
    print('Normalized Covariance-weighted Error: ',
          foo.computeError(G_results_ens))

    ## plot the results from G_results_ens and y_mean
    plt.plot(foo.traj_times, foo.y_mean, color='black')

    # loop over ensemble members
    for i in range(nSamples):
        plt.plot(foo.traj_times, G_results_ens[i], color='blue')

    # save the figure
    plt.savefig('data_vs_G.png')

