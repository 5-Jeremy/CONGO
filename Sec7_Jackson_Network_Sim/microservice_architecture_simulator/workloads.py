from microservice_architecture_simulator.registry import register_workload
import numpy as np
# Each workload function takes two required parameters, and then optionally accepts additional parameters
# The requried parameters are:
#   job_names: a list of the job names as given in the environment's yaml file; the weights returned by the function will be
#       assigned to the jobs in this order
#   time_step: the current time within the simulation, to allow for workloads that change depending on that time

@register_workload("single_job")
# Always send the specified job type
# Extra parameters:
#   active_job: The job type to send
def single_job_workload(job_names, round_num, active_job):
	dist = [0 for i in job_names]
	dist[job_names.index(active_job)] = 1
	return dist

@register_workload("static_distribution")
def static_distribution_workload(job_names, round_num, dist):
	return dist

### UNUSED ###
@register_workload("switching_single_job")
# Switch to the next job type in the predetermined schedule after every switching_period rounds
def switching_single_job_workload(job_names, round_num, switching_period, jobs_schedule):
	curr_job_indx = round_num // switching_period
	active_job_name = jobs_schedule[curr_job_indx % len(jobs_schedule)]
	active_job = job_names.index(active_job_name)
	dist = [0 for i in job_names]
	dist[active_job] = 1
	return dist

### UNUSED ###
@register_workload("random_jobs_switch")
# Pick num_jobs new job types at random after every switching_period rounds
# prev_dist is the job distribution from the previous round
def random_jobs_switch_workload(job_names, round_num, num_jobs, switching_period, prev_dist):
	if round_num == 0 or round_num % switching_period == 0:
		dist = np.zeros(len(job_names))
		active_jobs = np.random.randint(len(job_names), size=num_jobs)
		dist[active_jobs] = 1/num_jobs
		dist = dist.astype(np.float32)
		return dist
	return prev_dist

### UNUSED ###
@register_workload("uniform")
def uniform_workload(job_names, round_num):
	# Assign equal probability to all jobs
	num_types = len(job_names)
	dist = [1/num_types for i in job_names]
	return dist

@register_workload("gradual_switch")
# Gradually switch from exclusively one job type to exclusively a different job type
# Extra parameters:
#   time_horizon: how much time is given to transition between the job types
#   start_dist: The workload distribution used at the start
#   end_dist: The workload distribution used at the end
def gradual_switch_workload(job_names, round_num, start_round=None, end_round=None, start_dist=None, end_dist=None):
	start_dist_np = np.array(start_dist)
	end_dist_np = np.array(end_dist)
	ratio = min(max(round_num - start_round, 0)/(end_round - start_round), 1)
	dist_np = start_dist_np*(1 - ratio) + end_dist_np*ratio
	return dist_np.tolist()

### UNUSED ###
@register_workload("random")
def random_workload(job_names, round_num, change_freq = None ,prev_dist=None):
	if (round_num-1)%change_freq == 0:
		dist = np.random.random(len(job_names))
		dist /= dist.sum()
		dist = dist.astype(np.float32)
		return dist
	return prev_dist