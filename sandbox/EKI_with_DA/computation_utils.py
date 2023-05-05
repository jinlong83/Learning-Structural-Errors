#!/usr/bin/env python
import numpy as np
import pickle
# import io
import os
# from scipy.stats import gaussian_kde, entropy
# from statsmodels.tsa.stattools import acf
import pandas as pd
from pdb import set_trace as bp

def load_meals(fname):
	return np.array(pd.read_csv(fname))

def matt_xcorr(x, y):
	foo = np.correlate(x, y, mode='full')
	normalization = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
	xcorr = np.true_divide(foo,normalization)
	return xcorr

def linear_interp(x_vec, n_min, t, t0, dt):
	ind_mid = (t-t0) / dt
	ind_low = max(0, min( int(np.floor(ind_mid)), n_min) )
	ind_high = min(n_min, int(np.ceil(ind_mid)))
	v0 = x_vec[ind_low,:]
	v1 = x_vec[ind_high,:]
	tmid = ind_mid - ind_low
	return (1 - tmid) * v0 + tmid * v1


def replaceNaN(data):
	data[np.isnan(data)]=float('Inf')
	return data

def kde_scipy(x, x_grid, **kwargs):
	"""Kernel Density Estimation with Scipy"""
	# Note that scipy weights its bandwidth by the covariance of the
	# input data.  To make the results comparable to the other methods,
	# we divide the bandwidth by the sample standard deviation here.
	kde = gaussian_kde(x, **kwargs)
	return kde.evaluate(x_grid)

def kl4dummies(Xtrue, Xapprox, kde_func=kde_scipy, gridsize=1000):
	# arrays are identical and KL-div is 0
	if np.array_equal(Xtrue, Xapprox):
		return 0
	# compute KL-divergence
	x_grid, Pk, Qk = kdegrid(Xtrue, Xapprox, kde_func=kde_scipy, gridsize=gridsize)
	kl = entropy(Pk, Qk) # compute Dkl(P | Q)
	return kl

def kdegrid(Xtrue, Xapprox, kde_func=kde_scipy, gridsize=1000):
	zmin = min(min(Xtrue), min(Xapprox))
	zmax = max(max(Xtrue), max(Xapprox))
	x_grid = np.linspace(zmin, zmax, gridsize)
	Pk = kde_func(Xapprox.astype(np.float), x_grid) # P is approx dist
	Qk = kde_func(Xtrue.astype(np.float), x_grid) # Q is reference dist
	return x_grid, Pk, Qk

def computeErrors(target, prediction, dt, thresh):
	prediction = replaceNaN(prediction)

	# MEAN (over-space) SQUARE ERROR
	mse = np.mean((target-prediction)**2, axis=1)

	# TIME-AVG of MSE
	mse_total = np.mean(mse)

	# time convergence
	t_convergence = dt*getNumberOfBadPredictions(mse, thresh)

	eval_dict = {}
	for var_name in ['mse', 'mse_total', 't_convergence']:
		exec("eval_dict['{key}'] = {val}".format(key=var_name, val=var_name))
	return eval_dict


def getNumberOfBadPredictions(nerror, tresh=0.05):
	nerror_bool = nerror > tresh
	n_max = np.shape(nerror)[0]
	n = 0
	while nerror_bool[n] == True:
		n += 1
		if n == n_max: break
	return n

def getNumberOfAccuratePredictions(nerror, tresh=0.05):
	nerror_bool = nerror < tresh
	n_max = np.shape(nerror)[0]
	n = 0
	while nerror_bool[n] == True:
		n += 1
		if n == n_max: break
	return n
