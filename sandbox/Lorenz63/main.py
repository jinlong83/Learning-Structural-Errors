import numpy as np
import multiprocessing
from models import Lorenz63model 
from enki import EKI
import time
import pickle

def getData(state0, params):
    """
    Generate trajectory data
    state0: initial condition
    params: three parameters of Lorenz 63
    """
    model = Lorenz63model()
    T = 2.0
    dt = 0.01
    t_range = np.arange(0.0, T, dt)
    states = model.solve(params, state0, T, t_range, dt)   
    return states

def H(states):
    """
    Observation function
    Current observation: flattened 1-D data of all trajectories
    """
    obs = states.flatten()  
    return obs

def G(params_all):
    params = params_all[0:3]
    state0 = params_all[3:]
    model = Lorenz63model()
    T = 2.0
    dt = 0.01
    t_range = np.arange(0.0, T, dt)
    states = model.solve(params, state0, T, t_range, dt)
    return H(states)

def ensembleG(ui, parallel_flag = True):
    Gmatrix = np.zeros([ui.shape[0],600])
    if parallel_flag == True:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = []
        for iterN in range(ui.shape[0]):
            results.append(pool.apply_async(G, (ui[iterN,:],)))
        iterN = 0
        for result in results:
            Gmatrix[iterN,:] = result.get()
            iterN = iterN + 1
        pool.close()
    else:
        for iterN in range(ui.shape[0]):
            Gmatrix[iterN,:] = G(ui[iterN,:])
    return Gmatrix

if __name__ == "__main__":
    np.random.seed(10)
    start = time.time()

    ## Data generation with a initial condition
    state0 = np.array([1.,0.,0.])
    true_params = np.array([10.,28.,8./3.])
    y_mean = H(getData(state0, true_params))
    ## Clip the minimum std to 0.1
    y_cov = np.diag(np.clip((y_mean * 0.05)**2, 0.01, None))

    ## Settings of EKI and initial ensemble members
    DAsteps = 10    #EKI steps 
    nSamples = 100  #Ensemble size
    params_samples = np.zeros([nSamples,6])
    params_samples[:,0] = np.random.uniform(5, 20, nSamples) 
    params_samples[:,1] = np.random.uniform(25, 30., nSamples) 
    params_samples[:,2] = np.random.uniform(1, 5, nSamples) 
    params_samples[:,3] = np.random.uniform(0., 2., nSamples) 
    params_samples[:,4] = np.random.uniform(-1., 1., nSamples) 
    params_samples[:,5] = np.random.uniform(-1., 1., nSamples)   

    print("Number of cpu : ", multiprocessing.cpu_count())
    ## Initialize EKI object
    eki = EKI(params_samples, y_mean, y_cov, 1)
    ## Iterations of EKI steps
    for iterN in range(DAsteps):
        print('DA step: ', iterN+1)
        ## Forward simulation of ensemble members
        G_results = ensembleG(eki.u[iterN],True)
        ## Feed the ensemble evaluaation of G to EKI object
        eki.EnKI(G_results)
        print("Error: ", eki.error[-1])
        ## Save the current results of EKI
        pickle.dump(eki.u, open("u.pkl", "wb"))
        pickle.dump(eki.g, open("g.pkl", "wb"))
        pickle.dump(eki.error, open("error.pkl", "wb"))
        end = time.time()
        print('Time elapsed: ', end - start)
    ## Save the data for post-processing
    pickle.dump(y_mean, open("y_mean.pkl", "wb"))
    pickle.dump(y_cov, open("y_cov.pkl", "wb"))
