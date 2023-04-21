import os
import numpy as np
import multiprocessing
from models import Ultradian 
from utils import UnitGaussianNormalizer, MaxMinNormalizer
from enki import EKI
from time import time
import pickle
from pdb import set_trace as bp

DO_PARALLEL = True # parallelizes calls to G across ensemble members

OUTPUT_DIR = '.' # output directory

# Set a time range for data generation
NORMALIZER = MaxMinNormalizer
T_RANGE = np.arange(3500, 10000, 20.0) # generate data every 20 minutes
MAX_STEP = 1 # allow solver to take steps of up to 1 minute
DEFAULT_MODEL = Ultradian()
PARAM_NAMES = ['U0','Um','Rm'] # list of parameter names to be learned
TRUE_PARAMS = [getattr(DEFAULT_MODEL, param_name) for param_name in PARAM_NAMES]

# [print(f"{param_name} = {getattr(DEFAULT_MODEL, param_name)}") for param_name in PARAM_NAMES]


def getData(state0, params, param_names=PARAM_NAMES, t_range=T_RANGE, max_step=MAX_STEP):
    """
    Generate trajectory data
    state0: initial condition
    params: three parameters of Lorenz 63
    """
    model = Ultradian(param_names=param_names)
    states = model.solve(params, state0, t_range, max_step)   
    return states

def plotData(data, output_dir=OUTPUT_DIR):
    """
    Plot data
    """
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.savefig(os.path.join(output_dir, "data.pdf"))
    plt.close()

def H(states):
    """
    Observation function
    Current observation: flattened 1-D data of all trajectories
    """
    obs = states.flatten()  
    return obs

def G(params_all, normalizer, param_names=PARAM_NAMES, t_range=T_RANGE, max_step=MAX_STEP):
    n_params = len(param_names)
    params = params_all[:n_params]

    # map initial conditions to original model units
    state0 = normalizer.decode(params_all[n_params:])
    model = Ultradian(param_names=param_names)
    states = model.solve(params, state0, t_range, max_step)

    # Return observation of data in normalized units
    return H(normalizer.encode(states))

def ensembleG(ui, normalizer, parallel_flag = DO_PARALLEL, n_measurements=600):
    Gmatrix = np.zeros([ui.shape[0],n_measurements])
    if parallel_flag == True:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = []
        for iterN in range(ui.shape[0]):
            results.append(pool.apply_async(G, (ui[iterN,:], normalizer)))
        iterN = 0
        for result in results:
            Gmatrix[iterN,:] = result.get()
            iterN = iterN + 1
        pool.close()
    else:
        for iterN in range(ui.shape[0]):
            Gmatrix[iterN,:] = G(ui[iterN,:], normalizer)
    return Gmatrix

if __name__ == "__main__":
    output_dir = OUTPUT_DIR
    np.random.seed(10)
    start = time()

    true_params = TRUE_PARAMS #np.array([10.,28.,8./3.])

    ## Data generation with a initial condition
    state0 = np.array([85, 160, 10000])
    data = getData(state0, true_params)
    normalizer = NORMALIZER(data)
    plotData(normalizer.encode(data), output_dir)

    # print state0 in normalized units
    print("Initial condition in normalized units: ", normalizer.encode(state0)) 
    # note the normalization changes slightly with each noise realization

    y_mean = H(normalizer.encode(data))
    ## Clip the minimum std to 0.1
    y_cov = np.diag(np.clip((y_mean * 0.05)**2, 0.01, None))

    ## Settings of EKI and initial ensemble members
    DAsteps = 10    #EKI steps 
    nSamples = 100  #Ensemble size
    params_samples = np.zeros([nSamples,6])

    # PARAMS
    params_samples[:,:3] = np.array([DEFAULT_MODEL.sample_params(param_names=PARAM_NAMES) for _ in range(nSamples)])

    # STATES
    params_samples[:,3:] = np.array([normalizer.encode(DEFAULT_MODEL.sample_inits()) for _ in range(nSamples)])

    print("Number of cpu : ", multiprocessing.cpu_count())
    ## Initialize EKI object
    eki = EKI(params_samples, y_mean, y_cov, 1)
    ## Iterations of EKI steps
    for iterN in range(DAsteps):
        print('DA step: ', iterN+1)
        ## Forward simulation of ensemble members
        G_results = ensembleG(eki.u[iterN], normalizer, True, n_measurements=len(y_mean)) # nSamples x nData

        ## Feed the ensemble evaluaation of G to EKI object
        eki.EnKI(G_results)
        print("Error: ", eki.error[-1])
        ## Save the current results of EKI
        pickle.dump(eki.u, open(os.path.join(output_dir, "u.pkl"), "wb"))
        pickle.dump(eki.g, open(os.path.join(output_dir, "g.pkl"), "wb"))
        pickle.dump(eki.error, open(os.path.join(output_dir, "error.pkl"), "wb"))
        end = time()
        print('Time elapsed: ', end - start)
    ## Save the data for post-processing
    pickle.dump(y_mean, open(os.path.join(output_dir, "y_mean.pkl"), "wb"))
    pickle.dump(y_cov, open(os.path.join(output_dir, "y_cov.pkl"), "wb"))

