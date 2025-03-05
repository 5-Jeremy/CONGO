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
		env.settle_queues(action)
		cost,latency,done = env.step(action)
		per_env_step_cost_list.append(cost)
		per_round_cost_list.append(cost) 

		step_dir = []
		for i in range(len(action)):
			action_perturbed = action.copy()
			if action[i] - delta < 1e-3:
				action_perturbed[i] = action[i] + delta
				dir = 1
			elif action[i] + delta > max_alloc:
				action_perturbed[i] = action[i] - delta
				dir = -1
			else:
				dir = np.sign(np.random.binomial(1, 0.5) - 0.5)
				action_perturbed[i] = action[i] + dir*delta
			env.settle_queues(action)
			cost_perturbed,_,done = env.step(action_perturbed)
			per_env_step_cost_list.append(cost_perturbed)
			step_dir.append(dir*(cost - cost_perturbed))
		# step_dir tells us what direction we need to move in to decrease the cost, so step_dir_vector
			# is actually the opposite of the direction of the true gradient; hence we add it to get action_next
		step_dir_vector = np.array(step_dir)
		step_dir_vector = step_dir_vector/np.linalg.norm(step_dir_vector)
		if G['unstable']:
			# Try to escape the unstable region
			action_next = action + correction_factor
		else:
			action_next = action + step_dir_vector*lr
		action = project(action_next.copy())
		steps += 1
		env.reset_data()
		env.update_workload(steps)
	plt.plot(list(range(len(per_round_cost_list))), per_round_cost_list)
	plt.title("Naive Stochastic Gradient Descent")
	plt.xlabel("Round")
	plt.ylabel("Cost")
	plt.savefig(f"NSGD_{configuration_name}_{seed}.png")
	np.savetxt(f"NSGD_{configuration_name}_{seed}.csv", per_round_cost_list, delimiter=",")

if __name__ == "__main__":
	main()
