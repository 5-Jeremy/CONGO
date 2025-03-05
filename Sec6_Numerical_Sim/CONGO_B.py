from math import sqrt, ceil
from time import time_ns
import numpy as np
import numpy.linalg as la
from core_utils import *

from primal_dual import primal_dual_recovery_loose

# This is used for testing the effectiveness of the recovery algorithm under different conditions
from get_builtins import G
G['settings'] = None
G['gradient_errors'] = None

class CONGO_B(alg):
	def __init__(self, env, lr, m, delta=1e-5, k=None, init_point=None, rescale=True, enforce_gradient_limit=True):
		super().__init__(env, init_point)
		self.lr = lr
		self.delta = delta
		self.m = m
		if k is None:
			self.k = m
		else:
			self.k = k
		self.rescale = rescale
		self.enforce_gradient_limit = enforce_gradient_limit

	def step(self):
		self.costs.append(self.env.regret_sample(self.x))
		self.tot_costs.append(self.tot_costs[-1] + self.costs[-1])
		grad_est = self.estimate_gradient(self.x, self.env.info_sample, self.m, self.delta)
		self.x -= self.lr * grad_est
		self.x = self.env.project(self.x)
		self.t += 1

	def estimate_gradient(self, x, f, num_measurements, delta):
		m = num_measurements
		k = self.k
		# Generate the measurement matrix
		A = np.random.normal(0, 1, (m, x.shape[0]))
		# Perform the function evaluations
		y = np.zeros((m, k))
		Delta = np.random.choice([-1, 1], (m, k))
		perturbations = delta * A.T @ Delta
		base_value = f(x)
		for l in range(k):
			rescale_factor = la.norm(A.T @ Delta[:,l], 2)**2
			x_eval = x + np.expand_dims(perturbations[:,l], 1)/rescale_factor
			y[:,l] = (f(x_eval) - base_value)*rescale_factor/(delta * Delta[:,l])
		y_bar = np.sum(y, axis=1)/k
		if self.rescale:
			y_bar = y_bar/np.sqrt(m)
			A = A/np.sqrt(m)
		# Extra parameters for Chambolle and Pock's primal-dual algorithm
		theta = 1
		tau = 0.9/la.norm(A,2)
		sigma = tau
		grad_hat = primal_dual_recovery_loose(A, y_bar, 3*self.env.L*delta, theta, tau, sigma, 0.005, 50)
		if self.enforce_gradient_limit:
			if la.norm(grad_hat) > 3*self.env.L*delta + self.env.L_f:
				grad_hat *= 0
		return grad_hat
