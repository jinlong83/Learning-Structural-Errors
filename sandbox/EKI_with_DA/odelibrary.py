import numpy as np
from scipy.integrate import solve_ivp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

# from matplotlib import pyplot
# from numba import jitclass          # import the decorator
# from numba import boolean, int64, float32, float64    # import the types

from pdb import set_trace as bp
# Correspondence with Dima via Whatsapp on Feb 24, 2020:
# RK45 (explicit) for slow-system-only
# RK45 (implicit) aka Radau for multi-scale-system
# In both cases, set abstol to 1e-6, reltol to 1e-3, dtmax to 1e-3
# L96spec = [
#     ('K', int64),               # a simple scalar field
#     ('J', int64),               # a simple scalar field
#     ('hx', float64[:]),               # a simple scalar field
#     ('hy', float64),               # a simple scalar field
#     ('F', float64),               # a simple scalar field
#     ('eps', float64),               # a simple scalar field
#     ('k0', float64),               # a simple scalar field
#     ('slow_only', boolean),               # a simple scalar field
#     ('xk_star', float64[:])               # a simple scalar field
# ]

def my_solve_ivp(ic, f_rhs, t_eval, t_span, settings):
    u0 = np.copy(ic)
    if settings['method']=='Euler':
        dt = settings['dt']
        u_sol = np.zeros((len(t_eval), len(ic)))
        u_sol[0] = u0
        for i in range(len(t_eval)-1):
            t = t_eval[i]
            rhs = f_rhs(t, u0)
            u0 += dt * rhs
            u_sol[i] = u0
    else:
        # get settings not including 'dt' key
        settings = {k: v for k, v in settings.items() if k != 'dt'}
        sol = solve_ivp(fun=lambda t, y: f_rhs(t, y), t_span=t_span, y0=u0, t_eval=t_eval, **settings)
        u_sol = sol.y.T
    return np.squeeze(u_sol)

class UltradianGlucoseModel(object):
  """
  A simple class that implements Sturis and Polonsky Ultradian Glucose model

  The class computes RHS's to make use of scipy's ODE solvers.

  """

  def __init__(_s, eps=1e-5, constrain_positive=True, no_h=True, driver=np.zeros((0,2))):
    '''
    Initialize an instance: setting parameters and xkstar
    '''
     # set state to eps if it goes negative (non-physical).
    _s.eps = eps
    _s.constrain_positive = constrain_positive

    # remove last 3 states if no_h
    _s.no_h = no_h

    _s.Meals = driver # Time, carbs (mg)
    _s.Vp = 3 #'Vp' [l]
    _s.Vi = 11 #'Vi' [l]
    _s.Vg = 10 #'Vg' [l]
    _s.E = 0.2 #'E' [l min^-1]
    _s.tp = 6 #'tp' [min]
    _s.ti = 100 #'ti' [min]
    _s.td = 12 #'td' [min]
    _s.Rm = 209 #'Rm' [mU min^-1]
    _s.a1 = 6.67 #'a1' []
    _s.C1 = 300 #'C1' [mg l^-1]
    _s.C2 = 144 #'C2' [mg l^-1]
    _s.C3 = 100 #'C3' [mg l^-1]
    _s.C4 = 80 #'C4' [mU l^-1]
    _s.C5 = 26 #'C5' [mU l^-1]
    _s.Ub = 72 #'Ub' [mg min^-1]
    _s.U0 = 4 #'U0' [mg min^-1]
    _s.Um = 94 #'Um' [mg min^-1]
    _s.Rg = 180 #'Rg' [mg min^-1]
    _s.alpha = 7.5 #'alpha' []
    _s.beta = 1.77 #'beta' []
    _s.k_decay = 0.5 #'k_decay' []
    _s.kappa = (1/_s.Vi + 1/(_s.E*_s.ti))/_s.C4

    pmolperLpermU = 6.945 #conversion factor for insulin units (needed only when using Sturis)
    _s.adj_vec = np.array([pmolperLpermU/_s.Vp, pmolperLpermU/_s.Vi, 1/(10*_s.Vg), 1, 1, 1])

    _s.has_diverged = False

    # set state ranges
    _s.Ipmin, _s.Ipmax = (10, 300) # (80, 90)
    _s.Iimin, _s.Iimax = (10, 300) # (150, 170)
    _s.Gmin, _s.Gmax = (5000, 40000) # (9000, 11000)
    _s.h1min, _s.h1max = (10, 300) # (60, 80)
    _s.h2min, _s.h2max = (10, 300) # (60, 80)
    _s.h3min, _s.h3max = (10, 300) # (60, 80)

    _s.state_names = _s.get_state_names()

  def get_state_names(_s):
    if _s.no_h:
      return ['I_p', 'I_i', 'G', 'h_1', 'h_2', 'h_3']
    else:
      return ['I_p', 'I_i', 'G']


  def get_inits(_s):
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

  def F1(_s, G):
    return _s.Rm / ( 1 + np.exp(-G/(_s.Vg*_s.C1) + _s.a1) )

  def F2(_s, G):
      return _s.Ub*(1 - np.exp(-G/(_s.Vg*_s.C2)))

  def F3(_s, Ii):
    return ( _s.U0 + (_s.Um - _s.U0) / ( 1 + (_s.kappa*Ii)**(-_s.beta) ) ) / (_s.Vg* _s.C3)

  def F4(_s, h3):
    return _s.Rg/(1 + np.exp(_s.alpha*(h3/(_s.Vp*_s.C5)-1)))

  def SturisDynaPhenoExtendedMeals(_s, t_now):
    ## Exogenous glucose delivery rate function
    # from Albers Dynamical phenotyping paper
    # t_now is in MINUTES
    # Meals = [time(min),carbs(mg)]
    delrate = 0

    # initialize total current glucose delivery rate, del.
    if len(_s.Meals)==0:
      return delrate

    # find meal that occurred closest to t_now
    ind = np.argmin( np.abs(t_now- _s.Meals[:,0] ) )

    meal_before = _s.Meals[ind,0] <= t_now

    if meal_before or ind>0:
      if not(meal_before) and ind>0:
        ind -= 1
      for i in range(ind): #i=1:ind
        mstart = _s.Meals[i,0] # start time of ith meal
        #     I = Meals(i,3) # I constant for ith meal
        delrate += _s.Meals[i,1]*np.exp(_s.k_decay*(mstart - t_now)/60) # add ith meal contribution to del

    delrate = _s.k_decay * delrate / 60

    return delrate

  def rhs(_s, S, t):
    ## Sturis Ultradian Glucose Model from Keener Physiology Textbook

    ## Read in variables
    # force states to be positive
    if _s.constrain_positive:
      S[S < 0] = _s.eps

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




################################################################################
# end of ULTRADIAN ##################################################################
################################################################################



# @jitclass(L63spec)
class L63:
  """
  A simple class that implements Lorenz 63 model

  The class computes RHS's to make use of scipy's ODE solvers.

  Parameters:
    a, b, c

  """

  def __init__(_s,
      driver = None,
      a = 10, b = 28, c = 8/3, **kwargs):
    '''
    Initialize an instance: setting parameters and xkstar
    '''
    _s.a = a
    _s.b = b
    _s.c = c

    _s.state_names = ['x','y','z']


  def get_inits(_s):
    (xmin, xmax) = (-10,10)
    (ymin, ymax) = (-20,30)
    (zmin, zmax) = (10,40)

    xrand = xmin+(xmax-xmin)*np.random.random()
    yrand = ymin+(ymax-ymin)*np.random.random()
    zrand = zmin+(zmax-zmin)*np.random.random()
    state_inits = np.array([xrand, yrand, zrand])
    return state_inits

  def sample_params(_s, param_names):
    p_sample = np.zeros(len(param_names))
    for i, param_name in enumerate(param_names):
      p_val = getattr(_s, param_name)
      pmin = 0.5*p_val
      pmax = 2*p_val
      p_sample[i] = pmin+(pmax-pmin)*np.random.random()
    return p_sample

  def plot_state_indices(_s):
    return [0,1,2]

  def rhs(_s, S, t):
    ''' Full system RHS '''
    a = _s.a
    b = _s.b
    c = _s.c
    x = S[0]
    y = S[1]
    z = S[2]

    foo_rhs = np.empty(3)
    foo_rhs[0] = -a*x + a*y
    foo_rhs[1] = b*x - y - x*z
    foo_rhs[2] = -c*z + x*y

    return foo_rhs

################################################################################
# end of L63 ##################################################################
################################################################################
