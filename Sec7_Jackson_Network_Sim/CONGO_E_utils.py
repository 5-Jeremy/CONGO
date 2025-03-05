from math import sqrt, ceil
import numpy as np
import numpy.linalg as la
from microservice_architecture_simulator.envs.queue_utils import get_builtins

G = get_builtins()

import os, sys
sys.path.append(os.getcwd() + "/CONGO_Z_utils")
from CONGO_Z_utils.Cosamp import cosamp

def estimate_gradient_V3(x, L, resource_cost, num_measurements, delta, s):
		x = np.expand_dims(x, 1)
		m = num_measurements
		# Generate the measurement matrix
		A = np.random.normal(0, 1, (m, x.shape[0]))
		# Perform the function evaluations
		y = np.zeros((m, 1))
		latency_base = L(x)
		for i in range(m):
			x_pert = np.maximum(x + delta * np.expand_dims(A[i,:], 1), 1e-3)
			latency_pert = L(x_pert)
			y[i] = (latency_pert - latency_base)/delta
		y = y/np.sqrt(m)
		A = A/np.sqrt(m)
		
		grad_hat = cosamp(A, y.flatten(), s, 0.005, 50)
		grad_hat = grad_hat + resource_cost
		return grad_hat