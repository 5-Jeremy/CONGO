import hydra
import numpy as np
from microservice_architecture_simulator.registry import ENVS
from matplotlib import pyplot as plt
from microservice_architecture_simulator.envs.queue_utils import get_builtins
import os
import pyproximal as pyprox

G = get_builtins()

import sys
sys.path.append(os.getcwd() + "/CONGO_Z_utils")
from CONGO_Z_utils.optimizers import ZORO

def project(action):
	return np.clip(
			action,
			a_min=np.array([min_alloc]*num_nodes),
			a_max=np.array([max_alloc]*num_nodes),
		)

@hydra.main(config_path="conf", config_name="complex_env_fixed_workload", version_base=None)
def main(conf):
	global num_nodes, max_alloc, min_alloc
	configuration_name = conf.__dict__['_content']['exp']['name']
	print("Running configuration: ", configuration_name)
	env_class = ENVS[conf["env"]]
	env = env_class(conf["env_config"])
	seed = conf["env_config"]["seed"]
	num_nodes = len(conf["env_config"]["arch"]["initial_resource_alloc"])
	max_alloc = conf["env_config"]["arch"]["max_resources"]
	min_alloc = conf["env_config"]["arch"]["min_resources"]
	per_round_cost_list = []
	per_env_step_cost_list = []
	num_steps = 100 # steps is equivalent to rounds
	compute_nodes = list(conf["env_config"]["arch"]["initial_resource_alloc"].keys())
	action = np.array([conf["env_config"]["arch"]["initial_resource_alloc"][node][0] for node in compute_nodes],dtype=np.float32)
	steps = 0
	done = False
	correction_factor = conf["env_config"]["correction_factor"]
	delta = 0.5
	if configuration_name.startswith("complex_env"):
		s = 6
		m = 8
	elif configuration_name.startswith("large_scale"):
		s = 10
		m = 17
	### Initialize the ZORO algorithm
	def eval_latency(x):
		env.settle_queues(x)
		_,latency,done = env.step(x)
		return latency
	lr = 1.0 # Just a default value, this is updated later
	G['resource_weight'] = conf["env_config"]["reward"]["resource_weight"]
	prox = pyprox.Box(lower=np.array([min_alloc]*num_nodes), upper=np.array([max_alloc]*num_nodes))
	zoro_params = {"delta": delta, "sparsity": s, "step_size": lr, "num_samples": m}
	# We do not provide a function budget or a target value since we want to run for a fixed number of rounds
	zoro = ZORO(action, eval_latency, zoro_params, prox=prox)
	while steps < num_steps:
		G['unstable'] = False
		if configuration_name.startswith("complex_env"):
			if configuration_name.endswith("variable_job_type"):
				if steps == 0:
					lr = 0.7
				else:
					if steps % 25 == 0:
						lr *= 0.5
			else:
				if steps == 0:
					lr = 1.0
				else:
					if steps % 25 == 0:
						lr *= 0.7
		elif configuration_name.startswith("large_scale"):
			if steps == 0:
				if configuration_name.endswith("variable_job_type"):
					lr = 0.7
				else:
					lr = 1.0
		zoro.step_size = lr # Update the learning rate

		# Since the code for the ZORO algorithm provided by its creators handles function evaluations internally, we evaluate
		# the function at the previously chosen allocation here but do not count it towards the algorithm's budget for function 
		# evaluations.
		env.settle_queues(zoro.x)
		cost,latency,done = env.step(zoro.x)
		per_env_step_cost_list.append(cost)
		per_round_cost_list.append(cost)

		# Incase we need to recover from instability
		orig_x = zoro.x

		# Take a step with the ZORO algorithm; this includes estimating the gradient and choosing a new allocation
		zoro.step()

		# If instability occurs, override the choice made by the ZORO algorithm 
		if G['unstable']:
			# Try to escape the unstable region
			zoro.x = project(orig_x + correction_factor)

		steps += 1
		env.reset_data()
		env.update_workload(steps)
	plt.plot(list(range(len(per_round_cost_list))), per_round_cost_list)
	plt.title("CONGO-Z")
	plt.xlabel("Round")
	plt.ylabel("Cost")
	plt.savefig(f"CONGO-Z_{configuration_name}_{seed}.png")
	np.savetxt(f"CONGO-Z_{configuration_name}_{seed}.csv", per_round_cost_list, delimiter=",")

if __name__ == "__main__":
	main()
