# This script takes CSV files which it finds in the plotting directory and plots them together, assuming that they contain the costs 
	# observed on each round
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os

matplotlib.rcParams.update({'font.size': 14})
color_map = {"NSGD": 'teal', "SGDSP": 'green', "CONGO-B": 'blue', "CONGO-Z": 'purple', "CONGO-E": 'red'}
linestyle_map = {'NSGD': '-', 'SGDSP': '-.', 'CONGO-B': '--', 'CONGO-Z': ':', 'CONGO-E': '-.'}

# These paths should be relative to the directory from which this script will be run
data_dir = "/plotting_scripts/to_plot/"
fig_dir = "/plotting_scripts/to_plot/"

vertical_lines = None
labels = None
title = "Untitled"

# These were used to create the title and annotations for the plots in figure 4
#### Settings for 15 Queues, Fixed Workload
# title = "15 Queues, Fixed Workload"
##################################################
#### Settings for 15 Queues, Variable Arrival Rate
# title = "15 Queues, Variable Arrival Rate"
# vertical_lines = [0, 25, 50, 75]
# labels = ['5', '5.5', '6', '5.5']
# loc = 151
##################################################
#### Settings for 15 Queues, Variable Job Type
# title = "15 Queues, Variable Job Type"
# vertical_lines = [40, 90]
##################################################
#### Settings for 50 Queues, Fixed Workload
# title = "50 Queues, Fixed Workload"
##################################################
#### Settings for 50 Queues, Variable Arrival Rate
# title = "50 Queues, Variable Arrival Rate"
# vertical_lines = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# labels = ['4.5', '4.75', '5', '5.25', '5.5', '5.25', '5', '4.75', '4.5', '4.75']
# loc = 352
##################################################
#### Settings for 50 Queues, Variable Job Type
# title = "50 Queues, Variable Job Type"
# vertical_lines = [40, 90]
##################################################

# Define figure size
plt.figure(figsize=(8, 4))

if vertical_lines is not None:
	for vline in vertical_lines:
		plt.axvline(x=vline, color='gray', linestyle='--', linewidth=1)
if labels is not None:
	for vline, label in zip(vertical_lines, labels):
		plt.text(vline + 1, loc, label, rotation=0, verticalalignment='center', fontsize=12)

cost_trajectories = {'NSGD': [], 'SGDSP': [], 'CONGO-B': [], 'CONGO-Z': [], 'CONGO-E': []}

for data_file in os.listdir(os.getcwd() + data_dir):
	if not data_file.endswith('.csv'):
		continue
	data = np.genfromtxt(os.getcwd() + data_dir + data_file, delimiter=',')
	label = data_file.split('_')[0]
	if len(data.shape) > 1:
		data = data[:,1]
		# Note that we never used more than 10 seeds for a single experiment
	if len(data_file.split('_')[-1].split('.')[0]) != 1:
		plt.plot(list(range(len(data))), data, label=label, color=color_map[label])
	else:
		cost_trajectories[label].append(data)

for name in cost_trajectories.keys():
	if len(cost_trajectories[name]) == 0:
		continue
	stacked_trajectories = np.row_stack(cost_trajectories[name])
	mean_trajectory = stacked_trajectories.mean(axis=0)
	# print(mean_trajectory)
	stdev_trajectory = stacked_trajectories.std(axis=0)
	upper_stdev_trajectory = mean_trajectory + stdev_trajectory
	lower_stdev_trajectory = mean_trajectory - stdev_trajectory
	plt.plot(mean_trajectory, label=name, color=color_map[name], linestyle=linestyle_map[name])
	plt.fill_between(range(len(mean_trajectory)), lower_stdev_trajectory, upper_stdev_trajectory, alpha=0.1, color=color_map[name], linewidth=2)


plt.title(title, fontsize=24)
xlabel = "Round"
ylabel = "Cost"
plt.xlabel(xlabel, fontsize=22)
plt.ylabel(ylabel, fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlim(0, 100)
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.18, left=0.14, right=0.96)
# plt.legend(loc=(-0.1, 1.01), ncol=len(cost_trajectories.keys()), fontsize='small')
plt.savefig(os.getcwd() + fig_dir + title + '.png')