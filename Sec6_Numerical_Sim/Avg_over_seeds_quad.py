import numpy as np
import os

from core_utils import *
from gradient_descent import *
from GDSP import GDSP
from CONGO_B import CONGO_B
from CONGO_Z import CONGO_Z
from CONGO_E import CONGO_E

from math import ceil

# In this simulation, the cost function is generated randomly and takes the form
# x.T * D * x + x.T * b + c
# where D is a diagonal matrix where the nonzero entries are sparse and follow the absolute value of the standard 
# normal distribution, b is a sparse random vector drawn from the multivariate normal distribution, and c is the 
# absolute value of a scalar drawn from a Gaussian distribution.
# See core_utils.py for the definition of the Env and alg classes.
# The constraint set is a ball of radius R centered at the origin

device = 0

save_dir = 'trajectories/'
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

T = 100
R = 100
d = 50
s = 5
m = ceil(2 * s * np.log(d/s))
print("m = ", m)
delta = 1e-5
lr = 0.1

constraint_set = BallSet(device, d, R)

for seed in range(0,50):
	print("Begin seed ", seed)
	np.random.seed(seed)
	x0 = np.zeros((d,1))
	env = Env(device, d, s, constraint_set)
	alg_GD = GD(env, device, lr, init_point=x0)
	alg_GDSP = GDSP(env, device, lr, m, delta=delta, init_point=x0)
	alg_CONGO_B = CONGO_B(env, device, lr, m, delta=delta, init_point=x0, enforce_gradient_limit=True)
	alg_CONGO_Z = CONGO_Z(env, device, lr, m, delta=delta, init_point=x0, enforce_gradient_limit=True)
	alg_CONGO_E = CONGO_E(env, device, lr, m, delta=delta, init_point=x0, enforce_gradient_limit=True)
	for i in range(T):
		alg_GD.step()
		alg_GDSP.step()
		alg_CONGO_B.step()
		alg_CONGO_Z.step()
		alg_CONGO_E.step()
		env.new_cost_function()
	alg_GD.save_trajectory(save_dir + 'GD_seed_' + str(seed) + '.pkl')
	alg_GDSP.save_trajectory(save_dir + 'GDSP_seed_' + str(seed) + '.pkl')
	alg_CONGO_B.save_trajectory(save_dir + 'CONGO_B_seed_' + str(seed) + '.pkl')
	alg_CONGO_Z.save_trajectory(save_dir + 'CONGO_Z_seed_' + str(seed) + '.pkl')
	alg_CONGO_E.save_trajectory(save_dir + 'CONGO_E_seed_' + str(seed) + '.pkl')
