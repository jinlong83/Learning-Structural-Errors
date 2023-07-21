import numpy as np
import torch
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
        settings = {k: v for k, v in settings.items() if k not in ['dt', 'atol', 'rtol']}
        settings = {}
        sol = solve_ivp(fun=lambda t, y: f_rhs(t, y), t_span=t_span, y0=u0, t_eval=t_eval, **settings)
        u_sol = sol.y.T
    return np.squeeze(u_sol)

## NN functions
def create_nn(nn_dims, rescale_input=True, rescale_output=True):

    modules = []
    nn_num_layers = len(nn_dims) - 1
    if rescale_input:
      modules.append(RescaleLayer(nn_dims[0]))
      modules.append(torch.nn.Identity()) # need this because parameter fetcher uses every-other layer

    for idx in range(nn_num_layers):
        modules.append(torch.nn.Linear(nn_dims[idx], nn_dims[idx+1]))
        if idx < nn_num_layers - 1:
            modules.append(torch.nn.ReLU())
    
    if rescale_output:
      modules.append(torch.nn.Identity()) # need this because parameter fetcher uses every-other layer
      modules.append(RescaleLayer(nn_dims[-1]))

    net = torch.nn.Sequential(*modules)
    return net

# def my_nn(x, params, net): #, nn_num_layers):
def my_nn(params, net): #, nn_num_layers):
    '''This function first updates the network with specified parameters,
    then evaluates the network at the given input x'''
    nn_num_layers = int(np.ceil(len(net)/2))
    # print(int(np.ceil(len(net)/2))==nn_num_layers)
    # print(nn_num_layers)
    ## Error model
    params_start_idx = 0
    params_end_idx = 0
    for idx in range(nn_num_layers):
        layer_idx = int(idx*2) 
        ## Set i-th layer weights
        weight = net[layer_idx].weight
        params_num = weight.flatten().shape[0]
        params_end_idx += params_num
        net[layer_idx].weight = torch.nn.parameter.Parameter(torch.tensor(
                                params[params_start_idx:params_end_idx].reshape(weight.shape), 
                                dtype=torch.float32)) 
        params_start_idx = params_end_idx
        ## Set i-th layer bias 
        bias = net[layer_idx].bias 
        params_num = bias.flatten().shape[0]
        params_end_idx += params_num
        net[layer_idx].bias = torch.nn.parameter.Parameter(torch.tensor(
                              params[params_start_idx:params_end_idx].reshape(bias.shape),
                              dtype=torch.float32))
        params_start_idx = params_end_idx

        ##REMOVE ME: SET TO TRUTH MANUALLY
        # net[layer_idx].bias = torch.nn.Parameter(torch.tensor([0.0]))
        # net[layer_idx].weight = torch.nn.Parameter(torch.tensor([0, -0.01, 0], dtype=torch.float32).reshape(weight.shape))

    return net #(torch.tensor(x, dtype=torch.float32)).detach().numpy().flatten()

class RescaleLayer(torch.nn.Module):
    """ Custom normalization layer """
    # based on https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77

    def __init__(self, size_in):
        super().__init__()
        self.size_in = size_in
        weight = torch.Tensor(size_in)
        bias = torch.Tensor(size_in)
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

        # initialize weights and biases
        bound = 4 # log10
        torch.nn.init.uniform_(self.weight,  -bound, bound)  # weight init
        torch.nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w_times_x = torch.mul(x, 10**self.weight.t())
        return torch.add(w_times_x, 10**self.bias)  # 10^w times x + 10^b


## ODE functions


class UltradianGlucoseModel(object):
  """
  A simple class that implements Sturis and Polonsky Ultradian Glucose model

  The class computes RHS's to make use of scipy's ODE solvers.

  """

  def __init__(_s, eps=1e-5, constrain_positive=False, no_h=False,
               low_clip_nn=-1000/6,
               high_clip_nn=1000/6,
               keep_bounded=False,
               driver=np.zeros((0,2)), nn_dims=[3,1], nn_params=None,
               nn_rescale_input=False, nn_rescale_output=False,
               exptype='double',
               k_decay=0.5, a_decay=0.5, b_decay=10,
               Um=94, U0=4, Rm=209, Rg=180, ti=100, tp=6, Vg=10):
    '''
    Initialize an instance: setting parameters and xkstar
    '''
     # set state to eps if it goes negative (non-physical).
    _s.eps = eps
    _s.constrain_positive = constrain_positive
    _s.keep_bounded = keep_bounded

    # NN rescaling settings
    _s.nn_rescale_input = nn_rescale_input
    _s.nn_rescale_output = nn_rescale_output

    # remove last 3 states if no_h
    _s.no_h = no_h

    # clip neural net output
    _s.low_clip_nn = low_clip_nn
    _s.high_clip_nn = high_clip_nn

    _s.exptype = 'double' #'single' #exptype # 'single' or 'double' exponential meal function
    _s.Meals = driver # Time, carbs (mg)
    _s.Vp = 3 #'Vp' [l]
    _s.Vi = 11 #'Vi' [l]
    _s.Vg = Vg #0.1 for better scaling #10 for defaults #'Vg' [l]
    _s.E = 0.2 #'E' [l min^-1]
    _s.tp = tp #6 #'tp' [min]
    _s.ti = ti #100 #'ti' [min]
    _s.td = 12 #'td' [min]
    _s.Rm = Rm #209 #'Rm' [mU min^-1]
    _s.a1 = 6.67 #'a1' []
    _s.C1 = 300 #'C1' [mg l^-1]
    _s.C2 = 144 #'C2' [mg l^-1]
    _s.C3 = 100 #'C3' [mg l^-1]
    _s.C4 = 80 #'C4' [mU l^-1]
    _s.C5 = 26 #'C5' [mU l^-1]
    _s.Ub = 72 #'Ub' [mg min^-1]
    _s.U0 = U0 #4 #'U0' [mg min^-1]
    _s.Um = Um #94 #'Um' [mg min^-1]
    _s.Rg = Rg #180 #'Rg' [mg min^-1]
    _s.alpha = 7.5 #'alpha' []
    _s.beta = 1.77 #'beta' []
    _s.k_decay = k_decay #0.5 #'k_decay' []
    _s.a_decay = a_decay #0.5 #'k_decay' []
    _s.b_decay = b_decay #10 #'k_decay' []
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

    # set neural net
    _s.nn_params = nn_params
    _s.nn_dims = nn_dims # dimensions of each layer of the neural net
    # _s.nn_num_layers = len(_s.nn_dims) - 1 + _s.nn_rescale_input + _s.nn_rescale_output
    # print(_s.nn_dims, _s.nn_rescale_input, _s.nn_rescale_output)
    _s.net = create_nn(_s.nn_dims, rescale_input=_s.nn_rescale_input, rescale_output=_s.nn_rescale_output)
    _s.nn_num_params = sum(p.numel() for p in _s.net.parameters() if p.requires_grad)

  def set_nn(_s, nn_params):
    _s.my_nn = my_nn(nn_params, _s.net)

  def nn_eval(_s, x):
    with torch.no_grad():
    #   foo = my_nn(x, _s.nn_params, _s.net) #, _s.nn_num_layers)
        foo = _s.my_nn(torch.tensor(x, dtype=torch.float32)
                     ).detach().numpy().flatten()
    return foo

  def get_state_names(_s):
    if _s.no_h:
      return ['I_p', 'I_i', 'G']
    else:
      return ['I_p', 'I_i', 'G', 'h_1', 'h_2', 'h_3']


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

    print('WARNING: using true state inits w/ small noise')
    state_inits = np.array([50, 50, 10000, 10, 10, 10]) + np.random.randn()
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
    # if Ii < 0:
      # print('WARNING: Ii < 0')
      # Ii = 1e-6
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
        if _s.exptype=='single':
          delrate += _s.Meals[i,1]*np.exp(_s.k_decay*(mstart - t_now)/60) # add ith meal contribution to del
        elif _s.exptype=='double':
          delrate += _s.Meals[i,1]*(np.exp(_s.a_decay*(mstart - t_now)/60) - np.exp(_s.b_decay*(mstart -  t_now)/60))

    if _s.exptype=='single':
      delrate = _s.k_decay * delrate / 60
    elif _s.exptype=='double':
      delrate = _s.a_decay * _s.b_decay/(_s.b_decay-_s.a_decay) * delrate / 60

    return delrate

  def rhs(_s, S, t):
    # print(np.log10(S))
    ## Sturis Ultradian Glucose Model from Keener Physiology Textbook

    # this is the essential step for preventing the model from stalling
    # S = np.clip(S, a_min=[1e-5,1e-5,1e-5], a_max=[1e3, 1e3, 1e3])
    S = np.clip(S, a_min=[1e-5,1e-5,1e-5,1e-5,1e-5,1e-5], a_max=[1e4, 1e4, 1e6, 200, 200, 200])
    # this protects against big steps taken around meal time, causing untrue blowup.
    # this can protect against negative values of Ii, which can cause F3 to blowup.

    ## Read in variables
    # force states to be positive
    # if _s.constrain_positive:
    #   print('Constraining positive')
    #   is_neg = S < 0
    #   S[is_neg] = _s.eps

    # if _s.keep_bounded:
    #   print('bounding')
    #   bounds = [500, 500, 1e5]
    #   for idx in range(len(bounds)):
    #     if S[idx] > bounds[idx]:
    #       S[idx] = bounds[idx]

    if _s.no_h:
      Ip, Ii, G = S
    else:
      Ip, Ii, G, h1, h2, h3 = S

    # I am writing IG to be in mg/min, but should confirm that
    nutrition_oral = _s.SturisDynaPhenoExtendedMeals(t) #/ (10/_s.Vg)

    ## System of 6 ODEs##
    if _s.no_h:
      foo_rhs = np.zeros(3)  # a column vector
    else:
      foo_rhs = np.zeros(6)  # a column vector

    #plasma insulin mU
    foo_rhs[0] = _s.F1(G) - (Ip/_s.Vp - Ii/_s.Vi) * \
      _s.E - Ip/_s.tp  # dx/dt (mU/min)
    # print('tp=',_s.tp)
    # print('Ip/tp =', Ip/_s.tp)

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

    #  XXXXX add NN correction for insulin-dependent glucose absorption (Um=U0=0)
    # add NN correction for missing -I_p / t_p term in I_p equation (t_p=np.inf)
    if _s.nn_params is not None:
      # foo_rhs[1] -= np.abs(_s.nn_eval(S))
      # foo_rhs[0] += _s.nn_eval(S) 
      Smod = np.copy(S)
      Smod[2] /= 100 # scale G, since it is 2 orders of magnitude larger than other states
      # Smod /= 10
      # foo_rhs[0] += np.clip(_s.nn_eval(Smod), -100/6, 0)
      # foo_rhs[0] += _s.nn_eval(Smod)
      # foo_rhs[0] += np.clip(_s.nn_eval(Smod), -np.Inf, 0)
      # foo_rhs[0] += np.clip(_s.nn_eval(Smod), -1000/6, 1000/6)
      foo_rhs[0] += np.clip(_s.nn_eval(Smod), _s.low_clip_nn, _s.high_clip_nn)
      # foo_rhs[1] += _s.nn_eval(S)
      # print(_s.nn_eval(S))
      # bp()

    # if _s.constrain_positive:
    #   foo_rhs[is_neg] = np.abs(foo_rhs[is_neg])

    # clip foo_rhs to prevent overflow
    # if np.abs(foo_rhs[0]) > 250 or np.abs(foo_rhs[1]) > 250 or np.abs(foo_rhs[2]) > 5000:
    #   print(foo_rhs)

    # foo_rhs = np.clip(foo_rhs, a_min=1e-5*np.ones_like(foo_rhs), a_max=np.array([10,10,5000]))

    # print(foo_rhs)
    # print(t)
    # foo_rhs

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
      keep_bounded = True,
      a = 10, b = 28, c = 8/3,
      remove_y = False,
      nn_rescale_input = False, nn_rescale_output = False,
      nn_dims=[3,5,1], nn_params=None, 
      **kwargs):
    '''
    Initialize an instance: setting parameters and xkstar
    '''
    _s.a = a
    _s.b = b
    _s.c = c
    _s.keep_bounded = keep_bounded
    _s.state_names = ['x','y','z']

    # create model error
    _s.remove_y = remove_y

    # NN rescaling settings
    _s.nn_rescale_input = nn_rescale_input
    _s.nn_rescale_output = nn_rescale_output

    # set neural net
    _s.nn_params = nn_params
    _s.nn_dims = nn_dims # dimensions of each layer of the neural net
    # _s.nn_num_layers = len(_s.nn_dims) - 1 + _s.nn_rescale_input + _s.nn_rescale_output
    _s.net = create_nn(
        _s.nn_dims, rescale_input=_s.nn_rescale_input, rescale_output=_s.nn_rescale_output)
    _s.nn_num_params = sum(p.numel() for p in _s.net.parameters() if p.requires_grad)

  def set_nn(_s, nn_params):
    _s.my_nn = my_nn(nn_params, _s.net)

  def nn_eval(_s, x):
    with torch.no_grad():
    #   foo = my_nn(x, _s.nn_params, _s.net) #, _s.nn_num_layers)
        foo = _s.my_nn(torch.tensor(x, dtype=torch.float32)
                     ).detach().numpy().flatten()
    return foo

  def get_inits(_s):
    (xmin, xmax) = (-20,20)
    (ymin, ymax) = (-30,30)
    (zmin, zmax) = (5,45)

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
    if _s.keep_bounded:
      # print('keeping bounded')
      S[np.abs(S) > 200] = 200

    x = S[0]
    y = S[1]
    z = S[2]

    foo_rhs = np.empty(3)

    # X-component
    foo_rhs[0] = -a*x + a*y

    # Y-component
    if _s.remove_y:
      foo_rhs[1] = b*x - x*z
    else:
      foo_rhs[1] = b*x - y - x*z

    if _s.nn_params is not None:
      foo_rhs[1] += _s.nn_eval(S)

    # Z-component
    foo_rhs[2] = -c*z + x*y

    return foo_rhs

################################################################################
# end of L63 ##################################################################
################################################################################
