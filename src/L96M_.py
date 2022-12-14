import numpy as np

class L96M:
  """
  A simple class that implements Lorenz '96M model w/ slow and fast variables

  The class computes RHS's to make use of scipy's ODE solvers.

  Parameters:
    K, J, hx, hy, F, eps

  The convention is that the first K variables are slow, while the rest K*J
  variables are fast.
  """

  def __init__(_s,
      K = 36, J = 10, h = 10.0/1.0, F = 10, c = 1.0, b = 10.0, k0 = 0):
    '''
    Initialize an instance: setting parameters and xkstar
    '''
    hx = -h*c 
    hy = h*c/J 
    _s.b = b
    _s.c = c

    if not isinstance(hx, np.ndarray):
      hx = hx * np.ones(K)
    if hx.size != K:
      raise ValueError("'hx' must be a 1D-array of size 'K'")
    _s.predictor = None
    _s.K = K
    _s.J = J
    _s.hx = hx
    _s.hy = hy
    _s.F = F
    _s.k0 = k0 # for filtered integration
    _s.xk_star = 0.0 * np.zeros(K)
    _s.xk_star[0] = 5
    _s.xk_star[1] = 5
    _s.xk_star[-1] = 5

    _s.num_input = 1

  def set_params(_s, params):
    _s.params = params

  def set_params_Gamma(_s, params):
    _s.params_Gamma = params

  def set_num_input(_s, num_input):
    _s.num_input = num_input 

  def set_predictor(_s, predictor):
    _s.predictor = predictor

  def set_Gamma(_s, Gamma):
    _s.Gamma = Gamma 

  def set_G0_predictor(_s):
    _s.predictor = lambda x: _s.hy * x

  def set_stencil(_s, left = 0, right = 0):
    _s.stencil = np.arange(left, 1 + right)

  def hit_value(_s, k, val):
    return lambda t, z: z[k] - val

  def full(_s, t, z):
    ''' Full system RHS '''
    K = _s.K
    J = _s.J
    rhs = np.empty(K + K*J)
    x = z[:K]
    y = z[K:]

    ### slow variables subsystem ###
    # compute Yk averages
    Yk = _s.compute_Yk(z)

    # three boundary cases
    rhs[0] =   -x[K-1] * (x[K-2] - x[1]) - x[0]
    rhs[1] =   -x[0]   * (x[K-1] - x[2]) - x[1]
    rhs[K-1] = -x[K-2] * (x[K-3] - x[0]) - x[K-1]

    # general case
    rhs[2:K-1] = -x[1:K-2] * (x[0:K-3] - x[3:K]) - x[2:K-1]

    # add forcing
    rhs[:K] += _s.F

    # add coupling w/ fast variables via averages
    rhs[:K] += _s.hx * Yk

    ### fast variables subsystem ###
    # three boundary cases
    rhs[K]  = -y[1]  * (y[2] - y[-1]) * _s.b * _s.c - y[0] * _s.c 
    rhs[-2] = -y[-1] * (y[0] - y[-3]) * _s.b * _s.c - y[-2] * _s.c
    rhs[-1] = -y[0]  * (y[1] - y[-2]) * _s.b * _s.c - y[-1] * _s.c

    # general case
    rhs[K+1:-2] = -y[2:-1] * (y[3:] - y[:-3]) * _s.b * _s.c - y[1:-2] * _s.c

    # add coupling w/ slow variables
    for k in range(K):
      rhs[K + k*J : K + (k+1)*J] += _s.hy * x[k]

    return rhs

  def balanced(_s, t, x):
    ''' Only slow variables with balanced RHS '''
    K = _s.K
    rhs = np.empty(K)

    # three boundary cases: k = 0, k = 1, k = K-1
    rhs[0] = -x[K-1] * (x[K-2] - x[1]) - (1 - _s.hx[0]*_s.hy/_s.c) * x[0]
    rhs[1] = -x[0] * (x[K-1] - x[2]) - (1 - _s.hx[1]*_s.hy/_s.c) * x[1]
    rhs[K-1] = -x[K-2] * (x[K-3] - x[0]) - (1 - _s.hx[K-1]*_s.hy/_s.c) * x[K-1]

    # general case
    for k in range(2, K-1):
      rhs[k] = -x[k-1] * (x[k-2] - x[k+1]) - (1 - _s.hx[k]*_s.hy/_s.c) * x[k]

    # add forcing
    rhs += _s.F

    return rhs

  def regressed_odeint(_s, x, t):
    return _s.regressed(t, x)

  def regressed(_s, t, x):
    ''' Only slow variables with RHS learned from data '''
    K = _s.K
    rhs = np.empty(K)

    # three boundary cases: k = 0, k = 1, k = K-1
    rhs[0] = -x[K-1] * (x[K-2] - x[1]) - x[0]
    rhs[1] = -x[0] * (x[K-1] - x[2]) - x[1]
    rhs[K-1] = -x[K-2] * (x[K-3] - x[0]) - x[K-1]

    # general case
    for k in range(2, K-1):
      rhs[k] = -x[k-1] * (x[k-2] - x[k+1]) - x[k]

    # add forcing
    rhs += _s.F

    # add data-learned coupling term

    if _s.num_input == 1:
        R = _s.hx * _s.simulate(x)
    else:
        R = _s.hx * _s.simulate_nonlocal(x)

    rhs += R

    return rhs

  def fidx(_s, j, k):
    """Fast-index evaluation (based on the convention, see class description)"""
    return _s.K + k*_s.J + j

  def fidx_dec(_s, j, k):
    """Fast-index evaluation for the decoupled system"""
    return k*_s.J + j

  def simulate(_s, slow):
    predictor = _s.predictor(_s.apply_stencil(slow), _s.params)
    return np.reshape(predictor, (-1,))

  def simulate_nonlocal(_s, slow):
    num_input = _s.num_input
    shift = num_input // 2
    local_input = _s.apply_stencil(slow)
    all_input = np.roll(local_input, -shift, axis=0)
    for i in range(1, num_input):
        shifted_input = np.roll(local_input, i-shift, axis=0)
        all_input = np.hstack((all_input, shifted_input))
    predictor = _s.predictor(all_input, _s.params, num_input)
    return np.reshape(predictor, (-1,))

  def G(_s, slow, t):
    local_input = _s.apply_stencil(slow)
    all_input = np.roll(local_input, 0, axis=0)
    predictor = _s.Gamma(all_input, _s.params_Gamma)
    return np.diag(np.exp(np.reshape(predictor, (-1,))))

  def compute_Yk(_s, z):
    return z[_s.K:].reshape( (_s.J, _s.K), order = 'F').sum(axis = 0) / _s.J

  def gather_pairs(_s, tseries):
    n = tseries.shape[1]
    pairs = np.empty( (_s.K * n, _s.stencil.size + 1) )
    for j in range(n):
      pairs[_s.K * j : _s.K * (j+1), :-1] = _s.apply_stencil(tseries[:_s.K, j])
      pairs[_s.K * j : _s.K * (j+1), -1] = _s.compute_Yk(tseries[:,j])
    return pairs

  def gather_pairs_k0(_s, tseries):
    n = tseries.shape[1]
    pairs = np.empty( (n, 2) )
    for j in range(n):
      pairs[j, 0] = tseries[_s.k0, j]
      pairs[j, 1] = tseries[_s.K:, j].sum() / _s.J
    return pairs

  def apply_stencil(_s, slow):
    return slow[np.add.outer(np.arange(_s.K), _s.stencil) % _s.K]

################################################################################
# end of L96M ##################################################################
################################################################################
