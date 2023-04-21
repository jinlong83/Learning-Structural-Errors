import numpy as np
import scipy.integrate as scint
import pandas as pd
from pdb import set_trace as bp
# import warnings
# warnings.filterwarnings("error")


class Lorenz63model:

    def rhs(self, t, state):
        theta = self.theta
        x, y, z = state  # unpack the state vector
        dX = theta[0] * (y-x) 
        dY = x * (theta[1]-z) - y 
        dZ = x*y - theta[2]*z 
        return np.array([dX, dY, dZ])

    def solve(self, params, state0, t_range, dt):
        self.theta = params 
        sol = scint.solve_ivp(
                self.rhs,
                [t_range[0], t_range[-1]],
                state0,
                method = 'LSODA',
                t_eval = t_range,
                max_step = dt)
        states = sol.y.transpose()
        return states


class Ultradian:
    """
    A simple class that implements Sturis and Polonsky Ultradian Glucose model

    The class computes RHS's to make use of scipy's ODE solvers.

    Parameters:

    """

    def __init__(_s, eps=1e-5, no_h=True, param_names=[], nutrition_file='data/P1_nutrition_expert.csv'):
        '''
        Initialize an instance: setting parameters and xkstar
        '''
        # _s.eps = 1e-5  # set state to eps if it goes negative (non-physical).

        _s.no_h = no_h  # if True, then remove the final 3 states of the ODE

        # Load driver (nutrition sequence)
        try:
            driver = np.array(pd.read_csv(nutrition_file))
        except:
            Warning(f'Could not load driver from {nutrition_file}. Using empty default driver.')
            driver = np.zeros((0, 2))

        # set state names
        _s.state_names = ['I_p', 'I_i', 'G', 'h_1', 'h_2', 'h_3']
        if _s.no_h:
            _s.state_names = _s.state_names[:-3]

        # set names of learned parameters
        _s.param_names = param_names

        # set default parameters
        _s.Meals = driver  # Time, carbs (mg)
        _s.Vp = 3  # 'Vp' [l]
        _s.Vi = 11  # 'Vi' [l]
        _s.Vg = 10  # 'Vg' [l]
        _s.E = 0.2  # 'E' [l min^-1]
        _s.tp = 6  # 'tp' [min]
        _s.ti = 100  # 'ti' [min]
        _s.td = 12  # 'td' [min]
        _s.Rm = 209  # 'Rm' [mU min^-1]
        _s.a1 = 6.67  # 'a1' []
        _s.C1 = 300  # 'C1' [mg l^-1]
        _s.C2 = 144  # 'C2' [mg l^-1]
        _s.C3 = 100  # 'C3' [mg l^-1]
        _s.C4 = 80  # 'C4' [mU l^-1]
        _s.C5 = 26  # 'C5' [mU l^-1]
        _s.Ub = 72  # 'Ub' [mg min^-1]
        _s.U0 = 4  # 'U0' [mg min^-1]
        _s.Um = 94  # 'Um' [mg min^-1]
        _s.Rg = 180  # 'Rg' [mg min^-1]
        _s.alpha = 7.5  # 'alpha' []
        _s.beta = 1.77  # 'beta' []
        _s.k_decay = 0.5  # 'k_decay' []
        _s.kappa = (1/_s.Vi + 1/(_s.E*_s.ti))/_s.C4

        # conversion factor for insulin units (needed only when using Sturis)
        pmolperLpermU = 6.945
        _s.adj_vec = np.array(
            [pmolperLpermU/_s.Vp, pmolperLpermU/_s.Vi, 1/(10*_s.Vg), 1, 1, 1])

        _s.has_diverged = False

        # set state ranges
        _s.Ipmin, _s.Ipmax = (10, 300)  # (80, 90)
        _s.Iimin, _s.Iimax = (10, 300)  # (150, 170)
        _s.Gmin, _s.Gmax = (5000, 40000)  # (9000, 11000)
        _s.h1min, _s.h1max = (10, 300)  # (60, 80)
        _s.h2min, _s.h2max = (10, 300)  # (60, 80)
        _s.h3min, _s.h3max = (10, 300)  # (60, 80)

    def sample_inits(_s):

        Iprand = _s.Ipmin+(_s.Ipmax-_s.Ipmin)*np.random.random()
        Iirand = _s.Iimin+(_s.Iimax-_s.Iimin)*np.random.random()
        Grand  = _s.Gmin +(_s.Gmax-_s.Gmin)*np.random.random()
        h1rand = _s.h1min+(_s.h1max-_s.h1min)*np.random.random()
        h2rand = _s.h2min+(_s.h2max-_s.h2min)*np.random.random()
        h3rand = _s.h3min+(_s.h3max-_s.h3min)*np.random.random()
        state_inits = np.array([Iprand, Iirand, Grand, h1rand, h2rand, h3rand])
        if _s.no_h:
            state_inits = state_inits[:-3]

        return state_inits

    def sample_params(_s, param_names):
        p_sample = np.zeros(len(param_names))
        for i, param_name in enumerate(param_names):
            p_val = getattr(_s, param_name)
            pmin = 0.5*p_val
            pmax = 2*p_val
            p_sample[i] = pmin+(pmax-pmin)*np.random.random()
        return p_sample


    def set_params(_s, params):
        '''Set learned parameters'''
        for i, name in enumerate(_s.param_names):
            setattr(_s, name, params[i])

    def F1(_s, G):
        return _s.Rm / (1 + np.exp(-G/(_s.Vg*_s.C1) + _s.a1))

    def F2(_s, G):
        return _s.Ub*(1 - np.exp(-G/(_s.Vg*_s.C2)))

    def F3(_s, Ii):
        return (_s.U0 + (_s.Um - _s.U0) / (1 + (_s.kappa*Ii)**(-_s.beta))) / (_s.Vg * _s.C3)

    def F4(_s, h3):
        foo = _s.Rg/(1 + np.exp(_s.alpha*(h3/(_s.Vp*_s.C5)-1)))
        return foo

    def SturisDynaPhenoExtendedMeals(_s, t_now):
        ## Exogenous glucose delivery rate function
        # from Albers Dynamical phenotyping paper
        # t_now is in MINUTES
        # Meals = [time(min),carbs(mg)]
        delrate = 0

        # initialize total current glucose delivery rate, del.
        if len(_s.Meals) == 0:
            return delrate

        # find meal that occurred closest to t_now
        ind = np.argmin(np.abs(t_now - _s.Meals[:, 0]))

        meal_before = _s.Meals[ind, 0] <= t_now

        if meal_before or ind > 0:
            if not(meal_before) and ind > 0:
                ind -= 1
            for i in range(ind):  # i=1:ind
                mstart = _s.Meals[i, 0]  # start time of ith meal
                #     I = Meals(i,3) # I constant for ith meal
                # add ith meal contribution to del
                delrate += _s.Meals[i, 1]*np.exp(_s.k_decay*(mstart - t_now)/60)

        delrate = _s.k_decay * delrate / 60

        return delrate

    def rhs(_s, t, S):
        ## Sturis Ultradian Glucose Model from Keener Physiology Textbook

        ## Read in variables
        # force states to be positive
        # S[S < 0] = _s.eps

        if _s.no_h:
            Ip, Ii, G = S
        else:
            Ip, Ii, G, h1, h2, h3 = S

        # I am writing IG to be in mg/min, but should confirm that
        nutrition_oral = _s.SturisDynaPhenoExtendedMeals(t)

        ## System of 6 ODEs##
        if _s.no_h:
            foo_rhs = np.zeros(3)  # a column vector
        else:
            foo_rhs = np.zeros(6)  # a column vector

        #plasma insulin mU
        foo_rhs[0] = _s.F1(G) - (Ip/_s.Vp - Ii/_s.Vi) * \
            _s.E - Ip/_s.tp  # dx/dt (mU/min)

        #insterstitial insulin mU
        foo_rhs[1] = (Ip/_s.Vp - Ii/_s.Vi)*_s.E - Ii/_s.ti  # foo_rhs/dt (mU/min)

        #glucose in glucose space mg
        foo_rhs[2] = -_s.F2(G) - G*_s.F3(Ii) + nutrition_oral  # dz/dt (mg/min)
        if _s.no_h:
            foo_rhs[2] += _s.F4(Ip)
        #delay process h1,h2,h3
        else:
            foo_rhs[2] += _s.F4(h3)
            foo_rhs[3] = (Ip-h1)/_s.td
            foo_rhs[4] = (h1-h2)/_s.td
            foo_rhs[5] = (h2-h3)/_s.td

        return foo_rhs

    def solve(_s, params, state0, t_range, dt):
        _s.set_params(params) 
        sol = scint.solve_ivp(
                _s.rhs,
                [t_range[0], t_range[-1]],
                state0,
                method = 'LSODA',
                t_eval = t_range,
                max_step = dt)
        states = sol.y.transpose()
        return states
