import numpy as np
import numpy.linalg as la
from core_utils import *

from get_builtins import G

class GDSP(alg):
	def __init__(self, env, lr, k, delta=1e-5, init_point=None):
		super().__init__(env, init_point)
		self.lr = lr
		self.k = k
		self.delta = delta

	def step(self):
		self.costs.append(self.env.regret_sample(self.x))
		self.tot_costs.append(self.tot_costs[-1] + self.costs[-1])
		grad_estimates = np.zeros((self.env.d, 1))
		base_value = self.env.info_sample(self.x)
		for i in range(self.k):
			# U is a rademacher random variable which sets the random direction
			U = np.random.choice([-1, 1], (self.env.d, 1))
			grad_est = ((self.env.info_sample(self.x + self.delta*U) - base_value)/self.delta)*U
			if 'gradient_err_GDSP' in G.keys():
				G['gradient_err_GDSP'].append(la.norm(grad_est - self.env.curr_grad(self.x)))
			grad_estimates += grad_est
		grad_est = grad_estimates/self.k
		self.x -= self.lr * grad_est
		self.x = self.env.project(self.x)
		self.t += 1