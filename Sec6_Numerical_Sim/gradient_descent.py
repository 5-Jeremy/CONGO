import numpy as np
from core_utils import *

class GD(alg):
	def __init__(self, env, lr, init_point=None, mode='quadratic'):
		super().__init__(env, init_point)
		self.lr = lr
		self.mode = mode

	def step(self):
		self.costs.append(self.env.regret_sample(self.x))
		self.tot_costs.append(self.tot_costs[-1] + self.costs[-1])
		grad = self.env.curr_grad(self.x)
		self.x -= self.lr * grad
		self.x = self.env.project(self.x)
		self.t += 1