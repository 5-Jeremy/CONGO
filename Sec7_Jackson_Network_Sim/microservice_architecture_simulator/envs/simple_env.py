from microservice_architecture_simulator.registry import (
	register_env,
	create_workload,
)
from ray.rllib.env.env_context import EnvContext
import random
import gymnasium as gym
import numpy as np

import queueing_tool as qt
# Import the JobAgent class, along with all global variables used to configure it to the current workload
	# We use global variables instead of passing them as arguments to avoid needing to modify the queueing_tool library
from microservice_architecture_simulator.envs.queue_utils import get_builtins, JobAgent, arr_f

import time
# For quick list search
from itertools import count, filterfalse

# For sharing variables between modules
G = get_builtins()

# This class dictates how the Jackson network is set up, how the workload is generated, how the algorithm interacts with the
#	environment, and how the cost is calculated
@register_env("SimpleEnv")
class SimpleEnv(gym.Env):
# All jobs enter at a single node and go to a single output node when they are done being processed
# When we have jobs coming from multiple sources into a single node, we need them to all go into the same queue. To do this, we need a dummy node
# To collect them all first
# Thus, each microservice needs to be represented with an entry node, a queue edge, and an exit node. The connections between microservices will 
	# be edges without queues (service time set to 0)

	def __init__(self, config: EnvContext):
		# Initialize parameters
		self.epsilon = 0.001  # Lower limit to prevent divide by zero errors
		# Although we use this variable to keep track of the length of time for which the simulation has run,
			# we do not use it for anything since we plot in terms of round; note that for all algorithms except
			# SGDBO, a round requires the same amount of simulation time
		self.t = 0 
		self.T = config["T"] # Unused time horizon; the real horizon is in terms of rounds
		self.step_time = config["step_time"]
		self.settle_time = config["settle_time"]
		self.seed = config["seed"]
		# Usually we will set a fixed seed for reproducibility
		if self.seed != -1:
			random.seed(self.seed)
			np.random.seed(self.seed)
			queue_seed = self.seed
		else:
			queue_seed = random.randint(0, 1000000)
		self.config = config

		# Get objective function specs from config
		self.resource_weight = config["reward"]["resource_weight"]
		# Get workload specs from config
		self.workload = create_workload(config["jobs"]["workload"])
		self.workload_type = config["jobs"]["workload"]
		self.job_arrival_rate = config["jobs"]["poisson_arrival_rate"]
		self.job_paths = config["jobs"]["path"]
		self.job_types = list(self.job_paths.keys())

		# Pass additional parameters from the environment definition to the workload function if needed
		# The only workload types we actually used were "single_job", "static_distribution", and "gradual_switch"
		# See workloads.py for the explanation of each workload type
		if config["jobs"]["workload"] == "gradual_switch":
			self.workload_params = {
				"start_round": config["jobs"]["workload_params"]["start_round"],
				"end_round": config["jobs"]["workload_params"]["end_round"],
				"start_dist": config["jobs"]["workload_params"]["start_dist"],
				"end_dist": config["jobs"]["workload_params"]["end_dist"],
			}
		elif config["jobs"]["workload"] == "single_job":
			self.workload_params = {
				"active_job": config["jobs"]["workload_params"]["active_job"]
			}
		elif config["jobs"]["workload"] == "static_distribution":
			self.workload_params = {
				"dist": config["jobs"]["workload_params"]["dist"]
			}
		elif config["jobs"]["workload"] == "switching_single_job":
			self.workload_params = {
				"switching_period": config["jobs"]["workload_params"]["switching_period"],
				"jobs_schedule": config["jobs"]["workload_params"]["jobs_schedule"],
			}
		elif config["jobs"]["workload"] == "random_jobs_switch":
			self.workload_params = {
				"num_jobs": config["jobs"]["workload_params"]["num_jobs"],
				"switching_period": config["jobs"]["workload_params"]["switching_period"],
				"prev_dist": None,
			}
		elif config["jobs"]["workload"] == "random":
			# TODO: This has not been modified to work with the new system
			self.workload_params = {
				"change_freq":config["jobs"]["workload_params"]["change_freq"],
				"prev_dist": None,
			}
		else:
			self.workload_params = {}
		G['job_types'] = self.job_types
		self.update_workload(0) # Set initial workload

		self.compute_nodes = list(config["arch"]["initial_resource_alloc"].keys())
		self.max_resources = config["arch"]["max_resources"]
		self.min_resources = config["arch"]["min_resources"]

		#====# Initialize the queue network #====#
		# Anything stored in the dictionary G can be made accessible to functions in other files; this allows the JobAgent class
		# 	To change its behavior based on what we set here
		# The job routes define the path that each job type will take through the network
		G['job_routes'] = {job_type: ([node for node in path] + ["out"]) for job_type, path in self.job_paths.items()}
		### Assign numbers for each of the nodes/vertices that will be in the network
		# name2nodes_map is a mapping from a microservice to the corresponding input and output nodes. It also includes the in 
			# and out nodes, which do not represent microservices but rather are entry and exit points for the system
		name2nodes_map = {"in": [0]} # There is a single input node
		# Add the microservice nodes
		node_num = 1
		for ms in self.compute_nodes:
			name2nodes_map[ms] = [node_num, node_num + 1]
			node_num += 2
		name2nodes_map["out"] = [node_num]
		G['name2nodes_map'] = name2nodes_map
		### Next, create the adjacency list based on the known job routes. We first do this in terms of the microservice names
			# and then translate to node numbers
		ms_adja_list = {ms: [] for ms in self.compute_nodes}
		ms_adja_list["in"] = []
		# Keep track of the microservices which have an outbound connection, since those connections need to be monitored for 
		# 	jobs leaving the system
		outbound_microservices = [] 
		for path in self.job_paths.values():
			if path[0] not in ms_adja_list["in"]:
				ms_adja_list["in"].append(path[0])
			for i in range(len(path)-1):
				if path[i+1] not in ms_adja_list[path[i]]:
					ms_adja_list[path[i]].append(path[i+1])
			if "out" not in ms_adja_list[path[-1]]:
				ms_adja_list[path[-1]].append("out")
			if path[-1] not in outbound_microservices:
				outbound_microservices.append(path[-1])
		# Translate the microservice names to node numbers; we are considering connections from the output node of one microservice
			# to the input node of another microservice
		adja_list = {name2nodes_map[ms][1]: [name2nodes_map[ms2][0] for ms2 in ms_list] for ms, ms_list in ms_adja_list.items() if ms != "in"}
		# Add the input node's connections
		adja_list[0] = [name2nodes_map[ms][0] for ms in ms_adja_list["in"]]
		# Add the connections corresponding to the queues inside the microservices
		for ms in self.compute_nodes:
			adja_list[name2nodes_map[ms][0]] = [name2nodes_map[ms][1]]
		### Now, create the edge list specifying the type for each of the edges and the mapping from microservice names to 
		# 	queue edges which we will use to apply the correct service times based on the resource allocations
		# Due to a limitation of the current implementation, each queue edge must have a distinct type in order to have its 
		# 	service rate controlled independently
		edge_list = {0: {node: 1 for node in adja_list[0]}}
		name2queueEdge_map = {} 
		queue_num = 3
		for ms in self.compute_nodes:
			edge_list[name2nodes_map[ms][0]] = {name2nodes_map[ms][1]: queue_num}
			name2queueEdge_map[ms] = queue_num
			queue_num += 1
			edge_list[name2nodes_map[ms][1]] = {node: 2 for node in adja_list[name2nodes_map[ms][1]]}
		G['name2queueEdge_map'] = name2queueEdge_map
		### Prepare the dictionary of custom queue types to be passed into the initializer for the network
		q_classes = {c: qt.QueueServer for c in range(1, len(name2queueEdge_map.keys()) + 3)}
		g = qt.adjacency2graph(adjacency=adja_list, edge_type=edge_list)
		# If a dict is provided for the job arrival rate, it will specify on what rounds to change the arrival
		# 	rate and what the new rate will be each time
		if type(self.job_arrival_rate) == int:
			G['fixed_rate'] = self.job_arrival_rate
		else:
			G['fixed_rate'] = self.job_arrival_rate[0]
		# Setup the function which converts current allocation to service time
		self.alloc2serviceTime = lambda x: 1 / (x + 1e-1)
		G['service_times'] = {name2queueEdge_map[node]: self.alloc2serviceTime(config["arch"]["initial_resource_alloc"][node][0]) for node in self.compute_nodes}
		def service_func_factory(indx):
			def ser_f(t):
				# print("Service time for queue type {0} is {1}".format(indx, G['service_times'][indx]))
				return t + G['service_times'][indx]
			return ser_f
		entry_point_args = {
			1: {
			'arrival_f': arr_f,
			'service_f': lambda t: t, # Using this as the service time function means that the agent immediately leaves the queue
			'AgentFactory': JobAgent
			}
		}
		transfer_queue_args = {
			2: {'num_servers': 1, 'service_f': lambda t: t}
		}
		processing_queue_args = {
			i: {'num_servers': 1, 'service_f': service_func_factory(i)} for i in name2queueEdge_map.values()
		}
		
		q_args = {**entry_point_args, **processing_queue_args, **transfer_queue_args}
		self.qn = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args, seed=queue_seed)
		self.nodePair2edgeIndex_map = {queueserver.edge[0:2]: queueserver.edge[2] for queueserver in self.qn.edge2queue}
		G['nodePair2edgeIndex_map'] = self.nodePair2edgeIndex_map
		self.outbound_edges = [self.nodePair2edgeIndex_map[(name2nodes_map[ms][1], name2nodes_map["out"][0])] for ms in outbound_microservices]
		self.qn.initialize(edge_type=1) # Set the "entry point" queue, denoted by edge 1, to be active for arrivals
		### Setup data collection
		# Only collect data at edges connecting to the input or output, since that is all we need to calculate end-to-end latency
		# Currently, it is hardcoded so that all paths need to start with microservice A
		node_pair = (G['name2nodes_map']["in"][0], G['name2nodes_map']["A"][0])
		self.data_collection_queues = [self.nodePair2edgeIndex_map[node_pair]] + self.outbound_edges
		self.qn.start_collecting_data(queues=self.data_collection_queues)
		self.processed_job_ids = []
		self.first_unprocessed_indx = 0

	# This needs to be called periodically during a simulation so that the process of calculating latency does not get slowed 
	# 	down by extremely large lists
	def reset_data(self):
		self.qn.clear()
		self.qn.start_collecting_data(queues=self.data_collection_queues)
		self.qn.initialize(edge_type=1)
		self.processed_job_ids = []
		self.first_unprocessed_indx = 0
	
	# Update the workload at the end of an iteration
	def update_workload(self, round_num):
		G['job_probs'] = self.workload(self.job_types, round_num, **(self.workload_params))
		if self.workload_type == "random_jobs_switch" or self.workload_type == "random":
			self.workload_params["prev_dist"] = G['job_probs']
		if type(self.job_arrival_rate) != int:
			if round_num in self.job_arrival_rate.keys():
				G['fixed_rate'] = self.job_arrival_rate[round_num]
		
	def re_allocate_resources(self, action):
		# We assume that the projection has been done by the algorithm, but check it here
		# assert(np.all(action >= self.min_resources), "Resource allocation below minimum")
		# assert(np.all(action <= self.max_resources), "Resource allocation above maximum")
		action = action.reshape(
			len(self.compute_nodes),
			self.config["arch"]["num_resource_types"],
		)
		self.resource_alloc = action.astype(np.float32)
		G['service_times'] = {edge_indx:self.alloc2serviceTime(self.resource_alloc[self.compute_nodes.index(MS_name)]) for MS_name, edge_indx in G['name2queueEdge_map'].items()}

	def step(self, action):
		self.re_allocate_resources(action)
		self.t += self.step_time # Not used for anything
		# This advances the queueing network by the set amount of time, allowing new latencies to become available as jobs leave the system
		self.qn.simulate(t=self.step_time)
		# Get the logged data which includes entry and exit times for the jobs; see the queuing-tool documentation for details.
		# This includes all data that has been collected since the last time the system was reset, not just the data from the 
		# 	most recent simulation step; however, the data is stored in chronological order
		data = self.qn.get_agent_data(queues=self.data_collection_queues)
		data_keys = list(data.keys())
		### Determine the average end-to-end latency for jobs that reached the end of the network during this simulation step
		# First get the original end-to-end latencies
		latencies = {k:(data[k][-1,0] - data[k][0,0]) for k in data_keys[self.first_unprocessed_indx:] if (k not in self.processed_job_ids) and (data[k][-1,5] in self.outbound_edges)}
		# If a job has reached one of the edges which takes it out of the network, we record it as processed. That way, its
			# latency is not counted in future iterations
		self.processed_job_ids += [k for k in data_keys[self.first_unprocessed_indx:] if data[k][-1,5] in self.outbound_edges]
		# Keep track of the smallest index such that all job_ids less than that index have been processed
		self.first_unprocessed_indx = next(filterfalse(set(self.processed_job_ids[self.first_unprocessed_indx:]).__contains__, count(self.first_unprocessed_indx)))
		if len(latencies) > 0:
			avg_latency = np.mean(list(latencies.values()))
		else:
			avg_latency = np.nan # We don't want this to happen; we want jobs reaching the end of the network on every time step
			# If it does happen, need to take corrective action to make the system stable
			# print("No jobs reached the end of the network this time step")
			G["unstable"] = True
		cost = avg_latency + self.resource_weight*np.sum(self.resource_alloc)
		###
		terminated = self.t == self.T # This does not actually affect anything
		return (
			cost,
			avg_latency,
			terminated,
		)
	
	# This function is called after a change in resource allocation occurs so that the algorithm will wait for the end-to-end 
	# 	latency statistics to be affected before making measurements
	def settle_queues(self, action):
		self.re_allocate_resources(action)
		self.t += self.settle_time
		self.qn.simulate(t=self.settle_time)
		data = self.qn.get_agent_data()
		old_num_processed = len(self.processed_job_ids)
		self.processed_job_ids += [k for k in list(data.keys())[self.first_unprocessed_indx:] if data[k][-1,5] in self.outbound_edges]
		self.first_unprocessed_indx = next(filterfalse(set(self.processed_job_ids[self.first_unprocessed_indx:]).__contains__, count(self.first_unprocessed_indx)))
		new_num_processed = len(self.processed_job_ids)
		if new_num_processed == old_num_processed:
			# print("No jobs reached the end of the network during the settling period")
			pass