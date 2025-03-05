from math import sqrt, ceil
import numpy as np
import numpy.linalg as la
from microservice_architecture_simulator.envs.queue_utils import get_builtins

G = get_builtins()

def soft_threshold(x, _lambda):
	return np.sign(x) * np.maximum(np.abs(x) - _lambda, 0)

def elementwise_soft_threshold(x, _lambda):
	for i in range(np.shape(x)[0]):
		x[i] = soft_threshold(x[i], _lambda)
	return x

def primal_dual_recovery_loose(U, y, eta, theta, tau, sigma, gap_tol, max_iter):
	if len(y.shape) == 1:
		y = np.expand_dims(y, axis=1)
	x = U.T @ y
	x_bar = U.T @ y
	xi = np.zeros((U.shape[0],1))
	iter = 0
	gap = np.inf
	while iter < max_iter and gap > gap_tol:
		if la.norm(xi/sigma + U @ x_bar - y) <= eta:
			xi = np.zeros((U.shape[0],1))
		else:
			xi = (1 - eta*sigma/la.norm(xi + sigma*(U @ x_bar - y))) * (xi + sigma * (U @ x_bar - y))
		x_next = elementwise_soft_threshold(x - tau * (U.T @ xi), tau)
		x_bar = x_next + theta * (x_next - x)
		x = x_next
		iter += 1
		# Evaluate the duality gap
		if la.norm(U @ x - y) <= eta and np.max(np.abs(-U.T @ xi)) <= 1:
			gap = la.norm(x,1) + np.dot(np.squeeze(y), np.squeeze(xi)) + eta*la.norm(xi)
		else:
			gap = np.inf
	# if iter == max_iter:
		# print("Warning: Reached maximum number of iterations")
	return x

# L is the function which determines the average latency, and resource_cost is the cost per unit of resource allocated.
# Together, these can be used to determine the overall cost for any given allocation
def estimate_gradient(x, L, resource_cost, num_measurements, num_avg, delta, eta):
		x = np.expand_dims(x, 1)
		m = num_measurements
		k = num_avg
		# Generate the measurement matrix
		A = np.random.normal(0, 1, (m, x.shape[0]))
		# Perform the function evaluations
		y = np.zeros((m, k))
		Delta = np.random.choice([-1, 1], (m, k))
		perturbations = delta * A.T @ Delta
		base_latency = L(x)
		for l in range(k):
			x_eval = x + np.expand_dims(perturbations[:,l], 1)
			# Clip the perturbed values to ensure stability
			x_eval = np.maximum(x_eval, 1e-3)
			latency_eval = L(x_eval)
			y[:,l] = (latency_eval - base_latency)/(delta * Delta[:,l])
		y_bar = np.sum(y, axis=1)/k
		# The parameters for Chambolle and Pock's primal dual algorithm
		theta = 1
		tau = 0.9/la.norm(A,2)
		sigma = tau
		grad_hat = primal_dual_recovery_loose(A, y_bar, eta, theta, tau, sigma, 0.005, 50)
		# Add the contribution of the resource cost to the gradient
		grad_hat = grad_hat + resource_cost
		return grad_hat, 