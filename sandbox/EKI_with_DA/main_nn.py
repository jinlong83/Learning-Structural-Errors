import os
import numpy as np
import pandas as pd
from utils import UnitGaussianNormalizer, MaxMinNormalizer, InactiveNormalizer
from run_experiment_nn import EXPERIMENT

def run_all(MODELNAME='L63', meta_dir='results', seed=0):

    # set seed for reproducibility (only used for IC generation)
    np.random.seed(seed)

    # Global variables across experiments
    DO_PARALLEL = True  # parallelizes calls to G across ensemble members
    NORMALIZER = UnitGaussianNormalizer # normalizes observation data to zero mean, unit variance

    # Set to run NN inferences
    PARAM_TYPE = 'nn'

    ## L63 settings
    if MODELNAME == 'L63':
        DAsteps = 10 #EKI steps
        nSamples = 30 #Ensemble size

        DRIVER = None
        PARAM_NAMES = []  # list of parameter names to be learned
        DT = 0.05
        T_RANGE = np.arange(0, 15, DT)  # generate data every 20 minutes
        HOBS = np.array([[1, 0, 0]])
        T_DA = int(0.5*len(T_RANGE))
        IC_TRUE = np.array([10, 10, 10]) + np.random.randn(3)
        ODE_SETTINGS_APPROX = {'remove_y': True, 'nn_dims': [3,5,1], 
                               'nn_rescale_input': False, 
                               'nn_rescale_output': False}
        ODE_SETTINGS_TRUE = {'remove_y': False}
        
        DA_SETTINGS = {
            't0': T_RANGE[0],
            'H': HOBS,
            'dt': DT,
            'eta': 1.0,
            'add_state_noise': True, # seems like this needs to be True for EnKF to work well
            'state_noise_sd': 0.1,
            'obs_noise_sd': 1,
            'N_particles': 30, # only active if using enkf algorithm
            'integrator': 'RK45'
        }
    elif MODELNAME == 'UltradianGlucoseModel':
        DAsteps = 10    #EKI steps
        nSamples = 100 #20   #Ensemble size (need more particles for NN w/ 26 params)

        DRIVER = np.array(pd.read_csv('../../data/P1_nutrition_expert.csv'))
        PARAM_NAMES = [] # list of parameter names to be learned
        DT = 20.0 # data timestep (not an integration timestep)
        T_RANGE = np.arange(3500, 6000, DT) # np.arange(3500, 10000, DT) # generate data every 20 minutes
        # T_RANGE = np.arange(3500, 4500, DT) # np.arange(3500, 10000, DT) # generate data every 20 minutes
        T_DA = int(0.4*len(T_RANGE)) # number of measurements to use in assimilation warmup phase
        HOBS = np.eye(6) #np.array([[0, 0, 1]])
        IC_TRUE = np.array([50., 50., 10000, 10, 10, 10]) # + np.random.randn(3)
        # print('WARNING: using downscaled IC for UltradianGlucoseModel')
        # IC_TRUE = np.array([5, 5, 10])  # + np.random.randn(3)
        x_ic_mean = np.array([55., 130., 130.00, 10, 10, 10])
        # x_ic_mean = np.array([5.5, 13.0, 13.000])

        state_noise_cov = np.diag((0.01*(x_ic_mean)) ** 2)
        # ODE_SETTINGS_APPROX = {'Um': 0, 'U0': 0, 'nn_dims': [3, 25, 1]}
        ODE_SETTINGS_APPROX = {
                                # 'Rg': 0,
                               'tp': np.inf, 
                                'Vg': 10,
                                'no_h': False,
                               'nn_dims': [len(IC_TRUE), 5, 1], 
                            #    'constrain_positive': False,
                            #    'nn_dims': [3, 1], 
                                'nn_rescale_input': False,
                                'nn_rescale_output': False}
        ODE_SETTINGS_TRUE = {'no_h': False, 'Vg': 10}
        DA_SETTINGS = {
            't0': T_RANGE[0],
            'H': HOBS,
            'dt': DT,
            'eta': 1.0, # scale factor for 3DVAR gain matrix K = H.T / eta
            'add_state_noise': True, # add noise to generated dataset.
            'Sigma': state_noise_cov,
            'obs_noise_sd': 0.0001, #0.01,#20, # amount of noise to add to generated dataset
            'N_particles': 30, # only active if using enkf algorithm
            'integrator': 'RK45',
            # 'rtol': 0.01,
            # 'atol': 0.01,
        }

    ### Build EXPERIMENTS ###
    experiment = {}

    ## Experiment 1a: joint EKI ##
    experiment['joint'] = EXPERIMENT(
        seed=seed,
        param_type=PARAM_TYPE,
        use_da=False,
        output_dir=os.path.join(meta_dir,'partial_noisy_jointEKI'),
        nSamples=nSamples,
        DAsteps=DAsteps,
        parallel_flag=DO_PARALLEL,
        normalizer=NORMALIZER,
        dynamics_rhs=MODELNAME,
        param_names=PARAM_NAMES,
        ic_true=IC_TRUE,
        t_range=T_RANGE,
        dt=DT,
        t_da=0, # for JOINT, no need to discard transients since we infer the I.C. directly.
        driver=DRIVER,
        ode_settings_approx=ODE_SETTINGS_APPROX,
        ode_settings_true=ODE_SETTINGS_TRUE,
        da_settings=DA_SETTINGS)


    ## Experiment 2: 3dvar + EKI ##
    experiment['3dvar'] = EXPERIMENT(
        seed=seed,
        param_type=PARAM_TYPE,
        use_da=True,
        da_alg='3dvar',
        output_dir=os.path.join(meta_dir,'partial_noisy_3dvarEKI'),
        nSamples=nSamples,
        DAsteps=DAsteps,
        parallel_flag=DO_PARALLEL,
        normalizer=NORMALIZER,
        dynamics_rhs=MODELNAME,
        param_names=PARAM_NAMES,
        ic_true=IC_TRUE,
        t_range=T_RANGE,
        dt=DT,
        t_da=T_DA,
        driver=DRIVER,
        ode_settings_approx=ODE_SETTINGS_APPROX,
        ode_settings_true=ODE_SETTINGS_TRUE,
        da_settings=DA_SETTINGS)


    ## Experiment 3: EnKF + EKI ##
    experiment['enkf'] = EXPERIMENT(
        seed=seed,
        param_type=PARAM_TYPE,
        use_da=True,
        da_alg='enkf',
        output_dir=os.path.join(meta_dir,'partial_noisy_enkfEKI'),
        nSamples=nSamples,
        DAsteps=DAsteps,
        parallel_flag=DO_PARALLEL,
        normalizer=NORMALIZER,
        dynamics_rhs=MODELNAME,
        param_names=PARAM_NAMES,
        ic_true=IC_TRUE,
        t_range=T_RANGE,
        dt=DT,
        t_da=T_DA,
        driver=DRIVER,
        ode_settings_approx=ODE_SETTINGS_APPROX,    
        ode_settings_true=ODE_SETTINGS_TRUE,
        da_settings=DA_SETTINGS)

    ## Run experiments:
    # # run enkf
    # experiment['enkf'].run()

    # Run 3dvar
    # experiment['3dvar'].run()

    # run joint
    experiment['joint'].run()

    # run joint for longer
    # experiment['joint'].normalizer = NORMALIZER
    # experiment['joint'].nSamples = DA_SETTINGS['N_particles']*nSamples
    # experiment['joint'].output_dir = os.path.join(meta_dir,'partial_noisy_jointEKI_moreParticles')
    # experiment['joint'].run()


if __name__ == "__main__":
    # meta_dir = 'results_July4_2023_rcond1e-7/PartialObs_LowNoise_v1_tryRescale'
    # for seed in [1,2]:
    #     print('Running L63 experiments...')
    #     run_all('L63', meta_dir=os.path.join(
    #         meta_dir, f'L63_{seed}'), seed=seed)

    # meta_dir = 'results_NN_1stepahead_fixedt0Bug/run_tpInf_trueIC_fullObs_v2'
    meta_dir = 'results_Sep2_2023/FullObs_LowNoise_fixedICnotlearnt_tpInf_NN1.0/Daniel_debugging'
    for seed in [0,1]:
        print('Running Ultradian experiments...')
        run_all('UltradianGlucoseModel', meta_dir=os.path.join(
            meta_dir,f'UltradianGlucoseModel_{seed}_5hidden'), seed=seed)









