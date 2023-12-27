import os
import numpy as np
import pickle
from pydoc import locate

from computation_utils import computeErrors
from plotting_utils import *
from odelibrary import my_solve_ivp
from tqdm import tqdm
from pdb import set_trace as bp

class WRAPPER(object):
	def __init__(self,
				da_alg='3dvar',
				integrator='RK45',
				dt=0.1,
				dynamics_rhs=None,
				driver=None,
				da_settings={},
				ode_settings={},
				**kwargs):

		self.integrator_settings = {'dt': dt, 'method': integrator}

		if da_alg=='enkf':
			self.DA = ENKF(**da_settings)
		elif da_alg=='3dvar':
			self.DA = VAR3D(**da_settings)
		else:
			raise('DA algorithm not recognized.')

		## set up dynamics
		self.DA.integrator = integrator
		self.DA.ODE = locate('odelibrary.{}'.format(dynamics_rhs))
		self.DA.ode = self.DA.ODE(driver=driver, **ode_settings)

	def make_data(self, ic=None, t0=0, t_end=10000, step=5.0):

		traj_times = np.arange(start=t0, stop=t_end+step, step=step)
		if ic is None:
			ic = self.DA.ode.get_inits()
		t_span = [traj_times[0], traj_times[-1]]
		traj = my_solve_ivp(ic=ic, f_rhs=lambda t, y: self.DA.ode.rhs(y, t), t_eval=traj_times, t_span=t_span, settings=self.integrator_settings)

		# add noise and observation
		obs_noise_mean = np.zeros(self.DA.dim_y)
		obs_noise_cov = (self.DA.obs_noise_sd**2) * np.eye(self.DA.dim_y)
		traj_obs = (self.DA.H @ traj.T).T + np.random.multivariate_normal(mean=obs_noise_mean, cov=obs_noise_cov, size=len(traj_times))

		return traj_times, traj, traj_obs

	def solve(self, ic, times):
		# make predictions on the heldout data
		f_rhs = lambda t, y: self.DA.ode.rhs(y,t)
		t_eval = times
		t_span = [times[0], times[-1]]
		sol = my_solve_ivp(ic, f_rhs, t_eval, t_span, self.integrator_settings)
		return sol

	def G(self, obs, true, times, params, param_names, fig_path='G_run', make_plots=True):

		# reset output directory
		if make_plots:
			self.DA.output_dir = fig_path
			os.makedirs(self.DA.output_dir, exist_ok=True)

		# set parameters of the ODE
		for i, name in enumerate(param_names):
			setattr(self.DA.ode, name, params[i])

		# get initial condition
		self.DA.times = times
		self.DA.x_true = true
		self.DA.y_obs = obs
		self.DA.N = len(times)

		# Run filter to generate 1-step-ahead predictions
		self.DA.test_filter(make_plots=make_plots)

		# return 1-step ahead prediction
		return self.DA.y_pred

	# def get_nn(self, params):
	# 	'''return the neural network function defined by the given input parameters'''
	# 	# set parameters of the ODE
	# 	for i, name in enumerate(self.DA.ode.param_names):
	# 		setattr(self.DA.ode, name, params[i])
		
	# 	# return the neural network function
	# 	return self.DA.ode.nn_clipped_scaled_eval		

def generate_data(dt, t_eval, ode, t0=0):
	ic = ode.get_inits()
	t_span = [t_eval[0], t_eval[-1]]
	settings = {}
	settings['method'] = 'RK45'
	return my_solve_ivp(ic=ic, f_rhs=lambda t, y: ode.rhs(y, t), t_eval=t_eval, t_span=t_span, settings=settings)

class VAR3D(object):
	def __init__(self, H, output_dir='default_output_3DVAR',
				K=None, dt=0.01, T=100,
				# dynamics_rhs=None,
				integrator='RK45', t0=0, x_ic=None, obs_noise_sd=0.1,
				lr=0.005,
				driver=None,
				eta=1,
				**kwargs):

		# create output directory
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

		self.H = H # linear observation operator
		self.t0 = t0
		self.t_pred = t0
		self.t_assim = t0
		self.dt = dt
		self.T = T
		self.obs_noise_sd = obs_noise_sd
		self.lr = lr

		# set up observation data
		dim_y, dim_x = self.H.shape
		self.dim_y = dim_y
		self.dim_x = dim_x

		# obs_noise_mean = np.zeros(dim_y)
		# obs_noise_cov = (obs_noise_sd**2) * np.eye(dim_y)
		# self.y_obs = (self.H @ self.x_true.T).T + np.random.multivariate_normal(mean=obs_noise_mean, cov=obs_noise_cov, size=self.N)

		# set up useful DA matrices
		self.Ix = np.eye(dim_x)

		# set default gain to 0
		if K is None:
			K = eta * self.H.T #np.zeros((dim_x, dim_y)) # linear gain
		self.K = K


	def predict(self, ic, t0):
		t_span = [t0, t0+self.dt]
		t_eval = np.array([t0+self.dt])
		settings = {'dt': self.dt, 'method': self.integrator}
		foo = my_solve_ivp(ic=ic, f_rhs=lambda t, y: self.ode.rhs(y, t), t_eval=t_eval, t_span=t_span, settings=settings)
		return foo

	def update(self, x_pred, y_obs):
		return (self.Ix - self.K @ self.H) @ x_pred + (self.K @ y_obs)

	def test_filter(self, x_ic=None, make_plots=True):
		# set up DA arrays
		self.x_pred = np.zeros_like(self.x_true)
		self.y_pred = np.zeros_like(self.y_obs)
		self.x_assim = np.zeros_like(self.x_true)

		# choose ic for DA
		if x_ic is None:
			x_ic = self.ode.get_inits()
		self.x_assim[0] = x_ic
		self.x_pred[0] = x_ic
		self.y_pred[0] = self.H @ x_ic

		# DA @ c=0, t=0 has been initialized already
		self.t_pred = self.t0
		self.t_assim = self.t0
		for c in range(1, self.N):

			# predict
			self.x_pred[c] = self.predict(ic=self.x_assim[c-1], t0=self.t_pred)
			self.y_pred[c] = self.H @ self.x_pred[c]
			self.t_pred += self.dt

			# assimilate
			self.t_assim += self.dt
			self.x_assim[c] = self.update(x_pred=self.x_pred[c], y_obs=self.y_obs[c])

		# compute evaluation statistics on assimilation
		self.eval_dict_truth = computeErrors(target=self.x_true, prediction=self.x_assim, dt=self.dt, thresh=self.obs_noise_sd)

		# compute evaluation statistics on UNOBSERVED
		hidden_op = np.eye(self.dim_x) - self.H.T @ self.H
		self.eval_dict_hidden = computeErrors(target=(hidden_op@self.x_true.T).T, prediction=(hidden_op@self.x_assim.T).T, dt=self.dt, thresh=self.obs_noise_sd)

		# compute evaluation statistics on prediction
		self.eval_dict_obs = computeErrors(target=self.y_obs, prediction=self.y_pred, dt=self.dt, thresh=self.obs_noise_sd)

		# plot assimilation errors
		if make_plots:
			fig_path = os.path.join(self.output_dir, 'assimilation_errors_all')
			plot_assimilation_errors(times=self.times, errors=self.eval_dict_truth['mse'], eps=self.obs_noise_sd, fig_path=fig_path)

			fig_path = os.path.join(self.output_dir, 'hidden_errors_all')
			plot_assimilation_errors(times=self.times, errors=self.eval_dict_hidden['mse'], eps=self.obs_noise_sd, fig_path=fig_path)

			fig_path = os.path.join(self.output_dir, 'observation_errors_all')
			plot_assimilation_errors(times=self.times, errors=self.eval_dict_obs['mse'], eps=self.obs_noise_sd, fig_path=fig_path)

			# plot true vs assimilated trajectories
			for plot_pred in [0,1]:
				if plot_pred:
					pred = self.x_pred
				else:
					pred = None
				for burnin in [0, 100]:
					fig_path = os.path.join(self.output_dir, 'assimilation_traj_burnin{}_predplot{}'.format(burnin, plot_pred))
					plot_assimilation_traj(times=self.times, obs=self.y_obs, true=self.x_true, assim=self.x_assim, pred=pred, fig_path=fig_path, H=self.H, burnin=burnin, names=self.ode.state_names)


		self.x_assim_final = self.x_assim


class ENKF(object):
	def __init__(self, H, output_dir='default_output_EnKF',
				N_particles=10,
				obs_noise_sd=0.1,
				state_noise_sd=0,
				add_state_noise=False,
				Sigma=None,
				x_ic_mean=None,
				x_ic_sd=10,
				x_ic_cov=None,
				s_perturb_obs=True,
				dt=0.01, T=100,
				# dynamics_rhs=None,
				driver=None,
				integrator='RK45', t0=0,
				**kwargs):

		# create output directory
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

		self.N_particles = N_particles
		self.H = H # linear observation operator
		self.t0 = t0
		self.t_pred = t0
		self.t_assim = t0
		self.dt = dt
		self.T = T
		self.obs_noise_sd = obs_noise_sd
		self.state_noise_sd = state_noise_sd
		self.s_perturb_obs = s_perturb_obs
		self.add_state_noise = add_state_noise

		# set up observation data
		dim_y, dim_x = self.H.shape
		self.dim_y = dim_y
		self.dim_x = dim_x
		self.obs_noise_mean = np.zeros(dim_y)
		self.Gamma = (obs_noise_sd**2) * np.eye(dim_y) # obs_noise_cov
		# self.y_obs = (self.H @ self.x_true.T).T + np.random.multivariate_normal(mean=self.obs_noise_mean, cov=self.Gamma, size=self.N)

		# set up system process noise
		self.state_noise_mean = np.zeros(dim_x)
		if Sigma is None:
			self.Sigma = (state_noise_sd**2) * np.eye(dim_x)
		else:
			self.Sigma = Sigma

		self.x_ic_mean = x_ic_mean
		self.x_ic_sd = x_ic_sd
		self.x_ic_cov = x_ic_cov

	def predict(self, ic, t0):
		t_span = [t0, t0+self.dt]
		t_eval = np.array([t0+self.dt])
		settings = {'dt': self.dt, 'method': self.integrator}
		foo = my_solve_ivp(ic=ic, f_rhs=lambda t, y: self.ode.rhs(y, t), t_eval=t_eval, t_span=t_span, settings=settings)
		return foo

	def update(self, x_pred, y_obs):
		return (self.Ix - self.K @ self.H) @ x_pred + (self.K @ y_obs)

	def test_filter(self, x_ic_mean=None, x_ic_cov=None, x_ic_sd=None, make_plots=True):

		# set up DA arrays
		# means
		self.x_pred_mean = np.zeros_like(self.x_true)
		self.y_pred_mean = np.zeros_like(self.y_obs)
		self.x_assim_mean = np.zeros_like(self.x_true)

		# particles
		self.x_pred_particles = np.zeros( (self.N, self.N_particles, self.dim_x) )
		self.y_pred_particles = np.zeros( (self.N, self.N_particles, self.dim_y) )
		self.y_pred_particles_noiseFree = np.zeros( (self.N, self.N_particles, self.dim_y) )
		self.x_assim_particles = np.zeros( (self.N, self.N_particles, self.dim_x) )

		#  error-collection arrays
		self.x_assim_error_mean = np.zeros_like(self.x_true)
		self.x_pred_mean_error = np.zeros_like(self.x_true)
		self.y_pred_mean_error = np.zeros_like(self.y_obs)
		self.x_assim_error_particles = np.zeros( (self.N, self.N_particles, self.dim_x) )
		self.y_pred_error_particles = np.zeros( (self.N, self.N_particles, self.dim_y) )

		# cov
		self.x_pred_cov = np.zeros((self.N, self.dim_x, self.dim_x))

		# set up useful DA matrices
		self.Ix = np.eye(self.dim_x)
		self.K_vec = np.zeros( (self.N, self.dim_x, self.dim_y) )
		self.K_vec_runningmean = np.zeros_like(self.K_vec)

		# choose ic for DA
		if self.x_ic_cov is None:
			self.x_ic_cov = (self.x_ic_sd**2) * np.eye(self.dim_x)
		if self.x_ic_mean is None:
			self.x_ic_mean = np.zeros(self.dim_x)

		# generate initial ensemble using self.ode.get_inits()
		# x0 = np.random.multivariate_normal(mean=self.x_ic_mean, cov=self.x_ic_cov, size=self.N_particles)
		x0 = np.array([self.ode.get_inits() for _ in range(self.N_particles)])

		self.x_assim_particles[0] = np.copy(x0)
		self.x_pred_particles[0] = np.copy(x0)
		self.x_assim_mean[0] = np.mean(self.x_assim_particles[0], axis=0)
		self.x_pred_mean[0] = np.mean(self.x_pred_particles[0], axis=0)

		self.x_pred_mean[0] = np.mean(x0, axis=0)
		self.y_pred_mean[0] = self.H @ self.x_pred_mean[0]

		# DA @ c=0, t=0 has been initialized already
		for c in range(1, self.N):
			# compute and store ensemble forecasts
			for n in range(self.N_particles):
				self.x_pred_particles[c,n] = self.predict(ic=self.x_assim_particles[c-1,n], t0=self.t_pred)
				self.y_pred_particles_noiseFree[c,n] = self.H @ self.x_pred_particles[c,n]
				if self.add_state_noise:
					self.x_pred_particles[c,n] += np.random.multivariate_normal(mean=self.state_noise_mean, cov=self.Sigma, size=1).squeeze()
				self.y_pred_particles[c,n] = self.H @ self.x_pred_particles[c,n]

			## predict
			self.t_pred += self.dt

			# compute and store ensemble means
			self.x_pred_mean[c] = np.mean(self.x_pred_particles[c], axis=0)
			self.y_pred_mean[c] = self.H @ self.x_pred_mean[c]

			# track assimilation errors for post-analysis
			self.x_pred_mean_error[c] = self.x_true[c] - self.x_pred_mean[c]
			self.y_pred_mean_error[c] = self.H @ self.x_pred_mean_error[c]

			# compute and store ensemble covariance
			C_hat = np.cov(self.x_pred_particles[c], rowvar=False)
			self.x_pred_cov[c] = C_hat

			## compute gains for analysis step
			S = self.H @ C_hat @ self.H.T + self.Gamma
			self.K = C_hat @ self.H.T @ np.linalg.inv(S)
			self.K_vec[c] = np.copy(self.K)
			self.K_vec_runningmean[c] = np.mean(self.K_vec[:c], axis=0)

			## assimilate
			self.t_assim += self.dt
			for n in range(self.N_particles):
				# optionally perturb the observation
				y_obs_n = self.y_obs[c] + self.s_perturb_obs * np.random.multivariate_normal(mean=self.obs_noise_mean, cov=self.Gamma)

				# prediction error for the ensemble member
				self.y_pred_error_particles[c,n] = self.y_obs[c] - self.y_pred_particles[c,n]

				# update particle
				self.x_assim_particles[c,n] = self.update(x_pred=self.x_pred_particles[c,n], y_obs=y_obs_n)

				# track assimilation errors for post-analysis
				self.x_assim_error_particles[c,n] = self.x_true[c] - self.x_assim_particles[c,n]

			# compute and store ensemble means
			self.x_assim_mean[c] = np.mean(self.x_assim_particles[c], axis=0)

			# track assimilation errors for post-analysis
			self.x_assim_error_mean[c] = self.x_true[c] - self.x_assim_mean[c]

			# compute evaluation statistics
		self.eval_dict = computeErrors(target=self.x_true, prediction=self.x_assim_mean, dt=self.dt, thresh=self.obs_noise_sd)

		# plot assimilation errors
		if make_plots:
			fig_path = os.path.join(self.output_dir, 'assimilation_errors_all')
			plot_assimilation_errors(times=self.times, errors=self.eval_dict['mse'], eps=self.obs_noise_sd, fig_path=fig_path)

			# plot true vs assimilated trajectories
			for plot_pred in [0, 1]:
				if plot_pred:
					pred = self.x_pred_mean
				else:
					pred = None
				for burnin in [0, 100]:
					fig_path = os.path.join(
						self.output_dir, 'assimilation_traj_burnin{}_predplot{}'.format(burnin, plot_pred))
					plot_assimilation_traj(times=self.times, obs=self.y_obs, true=self.x_true, assim=self.x_assim_mean, pred=pred, fig_path=fig_path, H=self.H, burnin=burnin, names=self.ode.state_names)

			# plot K convergence
			fig_path = os.path.join(self.output_dir, 'K_sequence')
			plot_K_learning(times=self.times, K_vec=self.K_vec, fig_path=fig_path)

			fig_path = os.path.join(self.output_dir, 'K_runningMean')
			plot_K_learning(times=self.times, K_vec=self.K_vec_runningmean, fig_path=fig_path)

		# reassign means to universal variable names
		self.x_assim_final = self.x_assim_mean
		self.y_pred = np.mean(self.y_pred_particles_noiseFree, axis=1)
