import hydra
import numpy as np
from microservice_architecture_simulator.registry import ENVS
from matplotlib import pyplot as plt
from microservice_architecture_simulator.envs.queue_utils import get_builtins

G = get_builtins()

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
	action_next = action.copy()
	steps = 0
	done = False
	correction_factor = conf["env_config"]["correction_factor"]
	if configuration_name.startswith("simpler_env"):
		num_samples = 6
	elif configuration_name.startswith("complex_env"):
		num_samples = 8
	elif configuration_name.startswith("large_scale"):
		num_samples = 17
	delta = 0.5
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
		def eval_cost(x):
			env.settle_queues(x)
			cost,_,done = env.step(x)
			return cost
		
		first_cost = eval_cost(action)
		per_env_step_cost_list.append(first_cost)
		per_round_cost_list.append(first_cost)
		grad_estimates = np.zeros((num_nodes,))
		for i in range(num_samples):
			# U is a rademacher random variable which sets the random direction
			U = np.sign(np.random.binomial(1, 0.5, size=num_nodes) - 0.5)
			perturbed_cost = eval_cost(action + delta*U)
			grad_est = ((perturbed_cost - first_cost)/delta)*U
			grad_estimates += grad_est
		grad_dir_vector = grad_estimates/num_samples
		grad_dir_vector = grad_dir_vector/np.linalg.norm(grad_dir_vector)
		if G['unstable']:
			# Try to escape the unstable region
			action_next = action + correction_factor
		else:
			action_next = action - grad_dir_vector*lr
		action = project(action_next.copy())
		steps += 1
		
		env.reset_data()
		env.update_workload(steps)
	plt.plot(list(range(len(per_round_cost_list))), per_round_cost_list)
	plt.title("Stochastic Gradient Descent w/ Simultaneous Perturbation")
	plt.xlabel("Round")
	plt.ylabel("Cost")
	plt.savefig(f"SGDSP_{configuration_name}_{seed}.png")
	np.savetxt(f"SGDSP_{configuration_name}_{seed}.csv", per_round_cost_list, delimiter=",")

if __name__ == "__main__":
	main()
