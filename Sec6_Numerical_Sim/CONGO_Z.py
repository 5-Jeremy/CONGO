import numpy as np
from core_utils import *

from get_builtins import G

import sys, os
sys.path.append(os.getcwd() + "/ZORO_utils")
from ZORO_utils.optimizers import ZORO
import pyproximal as pyprox

class CONGO_Z(alg):
	def __init__(self, env, lr, k, delta = 1e-5, init_point=None, enforce_gradient_limit=True):
		super().__init__(env, init_point)
		self.lr = lr
		self.k = k
		self.delta = delta
		####
		prox = pyprox.EuclideanBall(center=0.0, radius=self.env.set.r)
		zoro_params = {"delta": self.delta, "sparsity": self.env.s, "step_size": lr, "num_samples": k}
		# We do not provide a function budget or a target value since we want to run for a fixed number of rounds
		self.zoro = ZORO(self.x, env.info_sample, zoro_params, prox=prox)
		# The gradient limit is a bound on the 1-norm of the matrix D times a constant (which we approximate with 11/2) times delta, plus the lipschitz constant
			# the bound on the 1-norm of the matrix D is stored in self.env.L
		if enforce_gradient_limit:
			self.zoro.gradient_limit = (7.21/2)*self.env.L*delta + self.env.L_f
		####

	def step(self):
		self.costs.append(self.env.regret_sample(self.zoro.x))
		self.tot_costs.append(self.tot_costs[-1] + self.costs[-1])
		self.zoro.step()
		self.t += 1