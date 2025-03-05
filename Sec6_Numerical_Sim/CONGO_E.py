from math import sqrt, ceil
from time import time_ns
import numpy as np
import numpy.linalg as la
from core_utils import *

from ZORO_utils.Cosamp import cosamp
from primal_dual import primal_dual_recovery_loose # Only used for comparison with CoSaMP

# This is used for testing the effect of m and the execution time of the different recovery algorithms
from get_builtins import G
G['settings'] = None
G['gradient_errors'] = None
G['CoSaMP_times'] = None
G['primal_dual_times'] = None

class CONGO_E(alg):
	def __init__(self, env, lr, m, delta=1e-5, init_point=None, rescale=True, enforce_gradient_limit=True):
		super().__init__(env, init_point)
		self.lr = lr
		self.delta = delta
		self.m = m
		self.rescale = rescale
		self.enforce_gradient_limit = enforce_gradient_limit

	def step(self):
		self.costs.append(self.env.regret_sample(self.x))
		self.tot_costs.append(self.tot_costs[-1] + self.costs[-1])
		# The gradient limit is a bound on the 2-norm of the matrix D times m times the maximum norm of a row of A times delta, plus the lipschitz constant
			# Since D is random diagonal matrix, we use a bound on the expectation of the maximum of the diagonal entries
			# Since the distribution for these entries is the absolute value of a normal distribution, we use the bound sqrt(2*ln(s))
		grad_est = self.estimate_gradient(self.x, self.env.info_sample, self.m, self.delta, enforce_gradient_limit=self.enforce_gradient_limit)
		self.x -= self.lr * grad_est
		self.x = self.env.project(self.x)
		self.t += 1

	def estimate_gradient(self, x, f, num_measurements, delta, enforce_gradient_limit=True):
		m = num_measurements
		# Generate the measurement matrix
		A = np.random.normal(0, 1, (m, x.shape[0]))
		# Perform the function evaluations
		y = np.zeros((m, 1))
		f_base = f(x)
		for i in range(m):
			row = np.expand_dims(A[i,:], 1)
			rowNormSqr = la.norm(row, 2)**2
			f_pert = f(x + delta * row/rowNormSqr)
			y[i] = (f_pert - f_base)*rowNormSqr/delta
		# Rescaling
		if self.rescale:
			y = y/np.sqrt(m)
			A = A/np.sqrt(m)
		######################################################
		# The first two branches were used for generating figures 5 and 6 in the paper
		if G['gradient_errors'] is not None:
			grad_hat_cosamp = cosamp(A, y.flatten(), self.env.s, 0.005, 50)
			grad_hat_cosamp = np.expand_dims(grad_hat_cosamp, axis=1)
			grad_err = la.norm(grad_hat_cosamp - self.env.curr_grad(x))
			G['gradient_errors'][m].append(grad_err)
			grad_hat = grad_hat_cosamp
		elif G['CoSaMP_times'] is not None:
			# Estimate the same gradient using CoSaMP and the primal-dual algorithm for basis pursuit
			start = time_ns()
			grad_hat_cosamp = cosamp(A, y.flatten(), self.env.s, 0.005, 50)
			end = time_ns()
			G['CoSaMP_times'].append((end - start)/1e6)
			start = time_ns()
			tau = 0.9/la.norm(A,2)
			grad_hat_primal_dual = primal_dual_recovery_loose(A, y, 1.5*self.env.L*delta, 1, tau, tau, 0.005, 50)
			end = time_ns()
			G['primal_dual_times'].append((end - start)/1e6)
			grad_hat = np.expand_dims(grad_hat_cosamp, axis=1)
		else:
			grad_hat = cosamp(A, y.flatten(), self.env.s, 0.005, 50)
			grad_hat = np.expand_dims(grad_hat, axis=1)
		if enforce_gradient_limit:
			if la.norm(grad_hat) > (7.21/2)*self.env.L*delta + self.env.L_f:
				grad_hat *= 0
		######################################################
		return grad_hat