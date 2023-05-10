import os
import numpy as np
import pandas as pd
from utils import UnitGaussianNormalizer, MaxMinNormalizer, InactiveNormalizer
from run_experiment import EXPERIMENT

def run_all(MODELNAME='L63', meta_dir='results', seed=0):

    # set seed for reproducibility (only used for IC generation)
    np.random.seed(seed)

    # Global variables across experiments
    DO_PARALLEL = True  # parallelizes calls to G across ensemble members
    NORMALIZER = InactiveNormalizer #UnitGaussianNormalizer # normalizes observation data to zero mean, unit variance

    ## L63 settings
    if MODELNAME == 'L63':
        DAsteps = 10 #EKI steps
        nSamples = 200 #Ensemble size

        DRIVER = None
        PARAM_NAMES = ['a', 'b', 'c']  # list of parameter names to be learned
        DT = 0.05
        T_RANGE = np.arange(0, 15, DT)  # generate data every 20 minutes
        HOBS = np.array([[1, 0, 0]])
        T_DA = int(0.5*len(T_RANGE))
        IC_TRUE = np.array([10, 10, 10]) + np.random.randn(3)
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
        nSamples = 20   #Ensemble size

        DRIVER = np.array(pd.read_csv('../../data/P1_nutrition_expert.csv'))
        PARAM_NAMES = ['U0','Um','Rm'] # list of parameter names to be learned
        DT = 20.0 # data timestep (not an integration timestep)
        T_RANGE = np.arange(3500, 6000, DT) # np.arange(3500, 10000, DT) # generate data every 20 minutes
        T_DA = int(0.4*len(T_RANGE)) # number of measurements to use in assimilation warmup phase
        HOBS = np.array([[0, 0, 1]])
        IC_TRUE = np.array([50, 50, 10000]) + np.random.randn(3)
        x_ic_mean = np.array([55, 130, 13000])
        state_noise_cov = np.diag((0.01*(x_ic_mean)) ** 2)
        DA_SETTINGS = {
            't0': T_RANGE[0],
            'H': HOBS,
            'dt': DT,
            'eta': 1.0, # scale factor for 3DVAR gain matrix K = H.T / eta
            'add_state_noise': True, # add noise to generated dataset.
            'Sigma': state_noise_cov,
            'obs_noise_sd': 20, # amount of noise to add to generated dataset
            'N_particles': 30, # only active if using enkf algorithm
            'integrator': 'RK45',
        }

    ### Build EXPERIMENTS ###
    experiment = {}

    ## Experiment 1a: joint EKI ##
    experiment['joint'] = EXPERIMENT(
        seed=seed,
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
        da_settings=DA_SETTINGS)


    ## Experiment 2: 3dvar + EKI ##
    experiment['3dvar'] = EXPERIMENT(
        seed=seed,
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
        da_settings=DA_SETTINGS)


    ## Experiment 3: EnKF + EKI ##
    experiment['enkf'] = EXPERIMENT(
        seed=seed,
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
        da_settings=DA_SETTINGS)

    ## Run experiments:
    experiment['3dvar'].run()

    # run enkf
    experiment['enkf'].run()

    # run joint
    experiment['joint'].run()

    # run joint for longer
    experiment['joint'].normalizer = NORMALIZER
    experiment['joint'].nSamples = DA_SETTINGS['N_particles']*nSamples
    experiment['joint'].output_dir = os.path.join(meta_dir,'partial_noisy_jointEKI_moreParticles')
    experiment['joint'].run()


if __name__ == "__main__":
    meta_dir = 'results_1stepahead_fixedt0Bug/run0'
    for seed in [1,2]:
        print('Running Ultradian experiments...')
        run_all('UltradianGlucoseModel', meta_dir=os.path.join(
            meta_dir,f'UltradianGlucoseModel_{seed}'), seed=seed)

    for seed in [1,2]:
        print('Running L63 experiments...')
        run_all('L63', meta_dir=os.path.join(
            meta_dir, f'L63_{seed}'), seed=seed)






