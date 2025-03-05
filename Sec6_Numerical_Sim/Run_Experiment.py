import numpy as np
import os
from math import ceil

from core_utils import *
from gradient_descent import *
from GDSP import GDSP
from CONGO_B import CONGO_B
from CONGO_Z import CONGO_Z
from CONGO_E import CONGO_E

# In this simulation, the cost function is generated randomly and takes the form
# x.T * D * x + x.T * b + c
# where D is a diagonal matrix where the nonzero entries are sparse and follow the absolute value of the standard 
# normal distribution, b is a sparse random vector drawn from the multivariate Gaussian distribution, and c is the 
# absolute value of a scalar drawn from a Gaussian distribution.
# See core_utils.py for the definition of the Env and alg classes.
# The constraint set is a ball of radius R centered at the origin

import argparse
parser = argparse.ArgumentParser()
# Input the experiment you want to run using the command line
parser.add_argument('exp_num', type=int, nargs=1)
args = parser.parse_args()
exp_num = vars(args)['exp_num'][0]

save_dir = 'trajectories/'
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

# Same for all experiments
T = 100
R = 100
lr = 0.1
n_seeds = 50

if exp_num == 1:
	d = 50
	s = 5
	delta = 1e-5
	m = ceil(2 * s * np.log(d/s))
	k_CONGO_B = 3*m
elif exp_num == 2:
	d = 50
	s = 5
	sigma = 0.001
	delta = 0.05
	m = ceil(2 * s * np.log(d/s))
	k_CONGO_B = 3*m
elif exp_num == 3:
	d = 100
	s = 5
	sigma = 0.001
	delta = 0.05
	m = ceil(2 * s * np.log(d/s))
	k_CONGO_B = 6*m
else:
	raise ValueError("Invalid choice for exp_num. Pick 1, 2, or 3.")

constraint_set = BallSet(d, R)

for seed in range(0,n_seeds):
	print("Begin seed ", seed)
	np.random.seed(seed)
	x0 = np.zeros((d,1))
	if exp_num == 1:
		env = Env(d, s, constraint_set)
	else:
		env = NoisyEnv(d, s, sigma, constraint_set)
	alg_GD = GD(env, lr, init_point=x0)
	alg_GDSP = GDSP(env, lr, m, delta=delta, init_point=x0)
	alg_CONGO_B = CONGO_B(env, lr, m, delta=delta, k=k_CONGO_B, init_point=x0, enforce_gradient_limit=True)
	alg_CONGO_Z = CONGO_Z(env, lr, m, delta=delta, init_point=x0, enforce_gradient_limit=True)
	alg_CONGO_E = CONGO_E(env, lr, m, delta=delta, init_point=x0, enforce_gradient_limit=True)
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

# Plot the results
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle

######### Formatting
matplotlib.rcParams.update({'font.size': 14})
# Default width, reduced height
figsize = (6.4, 4)
fig = plt.figure(figsize=(figsize[0], figsize[1]))
color_map = {'GD': 'saddlebrown', 'GDSP': 'g', 'CONGO_B': 'blue', 'CONGO_Z': 'purple', 'CONGO_E': 'red'}
linestyle_map = {'GD': '-', 'GDSP': '-.', 'CONGO_B': '--', 'CONGO_Z': ':', 'CONGO_E': '-.'}
plt.subplots_adjust(bottom=0.15)
plt.xlim(0, 100)
#########

plt.title(f'd = {d}, s = {s}, m = {m}, $\sigma = {sigma}$')
cost_trajectories = {'GD': [], 'GDSP': [], 'CONGO_B': [], 'CONGO_Z': [], 'CONGO_E': []}

for file in os.listdir('trajectories'):
	if file.endswith('.pkl'):
		with open('trajectories/' + file, 'rb') as f:
			data = pickle.load(f)
		if data['alg_name'] in cost_trajectories.keys():
			cost_trajectories[data['alg_name']].append(np.array(data['tot_costs']))

mean_trajectories = {}
for name in cost_trajectories.keys():
	stacked_trajectories = np.row_stack(cost_trajectories[name])
	mean_trajectories[name] = stacked_trajectories.mean(axis=0)
	stdev_trajectory = stacked_trajectories.std(axis=0)
	upper_stdev_trajectory = mean_trajectories[name] + stdev_trajectory
	lower_stdev_trajectory = mean_trajectories[name] - stdev_trajectory
	plt.plot(mean_trajectories[name], label=name, color=color_map[name], linestyle=linestyle_map[name])
	plt.fill_between(range(len(mean_trajectories[name])), lower_stdev_trajectory, upper_stdev_trajectory, alpha=0.1, color=color_map[name], linewidth=2)
plt.xlabel('Round')
plt.ylabel('Accumulated Cost')
plt.legend()

plt.savefig(f'exp{exp_num}.png')