import queueing_tool as qt
import numpy as np
import random

# Taken from https://stackoverflow.com/questions/4000141/how-can-i-create-global-classes-in-python-if-possible
def get_builtins():
	"""Due to the way Python works, ``__builtins__`` can strangely be either a module or a dictionary,
	depending on whether the file is executed directly or as an import. I couldn’t care less about this
	detail, so here is a method that simply returns the namespace as a dictionary."""
	return getattr( __builtins__, '__dict__', __builtins__ )

G = get_builtins()
G['G'] = G
G['job_probs'] = None
G['fixed_rate'] = None

class JobAgent(qt.Agent):
	def __init__(self, agent_id=(0, 0)):
		qt.Agent.__init__(self, agent_id)
		self.agent_id = self.agent_id[1]
		self.name2nodes_map = G['name2nodes_map']
		self.nodePair2edgeIndex_map = G['nodePair2edgeIndex_map']
		# On initialization, we decide which job this agent is going to represent; that is, we decide its route
		# Pick one of the job types according to the current probabilities
		job_type = random.choices(G['job_types'], G['job_probs'])[0]
		# print("Agent {0} is of type {1}".format(agent_id, job_type))

		# Put the route in terms of the node numbers (there is a single entry and exit node, and each microservice has a pair of nodes)
		self.route = [0]
		for node in G['job_routes'][job_type]:
			self.route += self.name2nodes_map[node]
		# Put the route in terms of the edge indices
		self.edge_route = [self.nodePair2edgeIndex_map[(self.route[i], self.route[i + 1])] for i in range(len(self.route) - 1)]
		self.edge_route.append(self.nodePair2edgeIndex_map[(self.route[-1], self.route[-1])])

	def __repr__(self):
		msg = "JobAgent; agent_id:{0}. time: {1}"
		return msg.format(self.agent_id, round(self._time, 3))

	def desired_destination(self, network, edge):
		"""Returns the agents next destination given their current
		location on the network.

		Parameters
		----------
		network : :class:`.QueueNetwork`
			The :class:`.QueueNetwork` where the Agent resides.
		edge : tuple
			A 4-tuple indicating which edge this agent is located at.
			The first two slots indicate the current edge's source and
			target vertices, while the third slot indicates this edges
			``edge_index``. The last slot indicates the edges edge
			type.

		Returns
		-------
		out : int
			Returns an the edge index corresponding to the agents next
			edge to visit in the network.
		"""
		curr_edge = edge[2]
		next_edge = self.edge_route[self.edge_route.index(curr_edge) + 1]
		return next_edge
	
def rate(t):
	return G['fixed_rate']

# Function called to sample inter-arrival times for jobs entering the system
def arr_f(t):
	# The last argument is the maximum value that rate can take
	return qt.poisson_random_measure(t, rate, G['fixed_rate'])