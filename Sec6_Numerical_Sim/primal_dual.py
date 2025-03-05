import numpy as np
import numpy.linalg as la

def soft_threshold(x, lmbda):
	return np.sign(x) * np.maximum(np.abs(x) - lmbda, 0)

def elementwise_soft_threshold(x, lmbda):
	for i in range(np.shape(x)[0]):
		x[i] = soft_threshold(x[i], lmbda)
	return x

# This modification is for example 15.7(b), which allows for bounded measurement error
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