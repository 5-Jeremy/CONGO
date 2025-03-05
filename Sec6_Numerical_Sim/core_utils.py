import numpy as np
import numpy.linalg as la
import pickle

class Env():
	def __init__(self, dims, sparsity, constraint_set=None):
		self.t = 0
		self.d = dims
		self.s = sparsity
		self.f = None
		# Since D is random diagonal matrix, we use a bound on the expectation of the maximum of the diagonal entries
			# Since the distribution for these entries is the absolute value of a normal distribution, we use the bound sqrt(2*ln(s))
		self.L = np.sqrt(2*np.log(self.s))
		# Lipschitz constant is a bound on the norm of D@x + b
			# We only use a bound on the expected value here to simplify things
		if type(constraint_set) == BallSet:
			# Since D is diagonal, the bound on the norm of D@x is maximum value which an entry of x can take
				# times the expected maximum entry of D
			self.L_f = constraint_set.r*np.sqrt(2*np.log(self.s)) + 2*np.sqrt(self.s)
		else:
			# Here the bound on the norm of D@x is the maximum value which an entry of x can take times the
				# expected norm of the diagonal of D (which is sqrt(s) since there are s nonzero values)
			self.L_f = max(np.abs(constraint_set.lb), np.abs(constraint_set.ub))*np.sqrt(self.s) + 2*np.sqrt(self.s)
		# print(f"Lipschitz constant: {self.L_f}")
		self.set = constraint_set
		self.new_cost_function()

	def new_cost_function(self):
		diag = np.zeros(self.d)
		self.b = np.zeros((self.d, 1))
		nonzero_entries = np.random.choice(self.d, size=(self.s,), replace=False)
		diag[nonzero_entries] = np.abs(np.random.randn(self.s))
		self.b[nonzero_entries] = np.random.normal(-1.0, 2.0, (self.s, 1))
		self.D = np.diag(diag)
		self.c = np.abs(np.random.normal(0.0, 1.0, (1,)))
	
	# We distinguish between samples which go towards the regret and samples used by the algorithm to ensure that the
		# calculated regret is not affected by noise when NoisyEnv is used
	def regret_sample(self, x):
		return (x.T @ self.D @ x + x.T @ self.b + self.c).squeeze()
	def info_sample(self, x):
		return self.regret_sample(x)
	
	# For a gradient descent baseline
	def curr_grad(self, x):
		return 2 * self.D @ x + self.b
	
	# For constrained optimization
	def project(self, x):
		if self.set is not None:
			return self.set.project(x)
		else:
			return x
		
	def constraint_violation(self, x):
		return la.norm(x - self.project(x))

class NoisyEnv(Env):
	def __init__(self, dims, sparsity, noise_bound, constraint_set=None):
		super().__init__(dims, sparsity, constraint_set)
		self.noise_bound = noise_bound

	def info_sample(self, x):
		return self.regret_sample(x) + np.random.normal(0.0, self.noise_bound, (1,))

class alg():
	def __init__(self, env, init_point=None):
		self.env = env
		self.t = 0
		if init_point is not None:
			self.x = init_point
		else:
			self.x = np.zeros((env.d, 1))
		self.costs = []
		self.tot_costs = [0]
		# self.constraint_violations = []
		# self.tot_constraint_violations = []

	def step(self):
		pass

	def save_trajectory(self, filename):
		save_data = {'alg_name': self.__class__.__name__, \
					'tot_costs': np.array(self.tot_costs),\
					'dims': self.env.d,\
					'sparsity': self.env.s}
		with open(filename, 'wb') as f:
			pickle.dump(save_data, f)
	
	def save_trajectory_queue(self, filename):
		save_data = {'alg_name': self.__class__.__name__, \
					'tot_costs': np.array(self.tot_costs),\
					'dims': self.env.d,\
					'constraint_set': self.env.set}
		with open(filename, 'wb') as f:
			pickle.dump(save_data, f)

class ConstraintSet():
	def __init__(self, dims):
		self.d = dims
	
	def project(self, x):
		pass

class BallSet(ConstraintSet):
	def __init__(self, dims, radius):
		super().__init__(dims)
		self.r = radius

	def project(self, x):
		return x/max(1.0, la.norm(x)/self.r)

class HyperCubeSet(ConstraintSet):
	def __init__(self, dims, lower_bound, upper_bound):
		super().__init__(dims)
		self.lb = lower_bound
		self.ub = upper_bound

	def project(self, x):
		return np.clip(x, self.lb, self.ub)