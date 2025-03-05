import hydra
import numpy as np
from microservice_architecture_simulator.registry import ENVS
from matplotlib import pyplot as plt
from microservice_architecture_simulator.envs.queue_utils import get_builtins

G = get_builtins()

from CONGO_E_utils import estimate_gradient_V3

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
	num_steps = 100 # steps is equivalent to rounds
	compute_nodes = list(conf["env_config"]["arch"]["initial_resource_alloc"].keys())
	action = np.array([conf["env_config"]["arch"]["initial_resource_alloc"][node][0] for node in compute_nodes],dtype=np.float32)
	action_next = action.copy()
	steps = 0
	done = False
	correction_factor = np.array(conf["env_config"]["correction_factor"])
	delta = 0.5
	if configuration_name.startswith("complex_env"):
		s = 6
		m = 8
	elif configuration_name.startswith("large_scale"):
		s = 10
		m = 17
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
		
		def eval_latency(x):
			env.settle_queues(x)
			_,latency,done = env.step(x)
			return latency
		resource_weight = conf["env_config"]["reward"]["resource_weight"]

		# This evaluation is for plotting
		per_round_cost_list.append(eval_latency(action) + np.sum(resource_weight*action))

		grad_hat = estimate_gradient_V3(action, eval_latency, resource_weight, m, delta, s)
		
		if G['unstable']:
			# Try to escape the unstable region
			action_next = action + correction_factor
		else:
			# We have found that for all the gradient-descent base algorithms considered, normalization of the gradient to a unit vector
			# helps with robustness under the dynamics of a realistic queueing system
			grad_hat = grad_hat/np.linalg.norm(grad_hat)
			grad_hat = grad_hat.reshape(-1)
			action_next = action - grad_hat*lr
		
		action = project(action_next.copy())
		steps += 1

		env.reset_data()
		env.update_workload(steps)
	plt.plot(list(range(len(per_round_cost_list))), per_round_cost_list)
	plt.title("CONGO-E")
	plt.xlabel("Round")
	plt.ylabel("Cost")
	plt.savefig(f"CONGO-E_{configuration_name}_{seed}.png")
	np.savetxt(f"CONGO-E_{configuration_name}_{seed}.csv", per_round_cost_list, delimiter=",")

if __name__ == "__main__":
	main()
