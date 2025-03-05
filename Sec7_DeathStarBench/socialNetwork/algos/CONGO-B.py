import random
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import os
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import linprog
from scipy.stats import qmc  # Import for quasi-random sequence generation

#============================
### GLOBAL VARIABLES DECLARATION
#============================
# Weights for cost calculation
latency_weight = 1.0
cpu_allocation_weight = 10.0
num_of_measurements = 11

# Container names for the experiment
container_names = [
    "socialnetwork-user-service-1",
    "socialnetwork-post-storage-service-1",
    "socialnetwork-media-service-1",
    "socialnetwork-social-graph-service-1",
    "socialnetwork-user-timeline-service-1",
    "socialnetwork-url-shorten-service-1",
    "socialnetwork-text-service-1",
    "socialnetwork-unique-id-service-1",
    "socialnetwork-nginx-thrift-1",
    "socialnetwork-media-frontend-1",
    "socialnetwork-compose-post-service-1",
    "socialnetwork-home-timeline-service-1",
    "socialnetwork-user-mention-service-1",
    "socialnetwork-user-mongodb-1",
    "socialnetwork-social-graph-redis-1",
    "socialnetwork-home-timeline-redis-1",
    "socialnetwork-social-graph-mongodb-1",
    "socialnetwork-url-shorten-memcached-1",
    "socialnetwork-post-storage-mongodb-1",
    "socialnetwork-user-timeline-redis-1",
    "socialnetwork-jaeger-agent-1",
    "socialnetwork-user-timeline-mongodb-1",
    "socialnetwork-user-memcached-1",
    "socialnetwork-post-storage-memcached-1",
    "socialnetwork-media-mongodb-1",
    "socialnetwork-media-memcached-1",
    "socialnetwork-url-shorten-mongodb-1"
]

num_of_containers = len(container_names)


def execute_docker_command(cmd):
    """Executes a Docker command and returns its output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def update_container_resources(container_name, cpu_allocation, memory_allocation):
    """Update the CPU and memory allocation for a given container."""
    cpu_period = 100000
    cpu_quota = int(cpu_allocation * cpu_period)
    hardcoded_password = "your_placeholder_password"  # Placeholder for actual password
    if memory_allocation != -1:
        cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpus={cpu_allocation} --memory={memory_allocation}m --memory-swap={memory_allocation}m {container_name}'
    else:
        cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpu-period={cpu_period} --cpu-quota={cpu_quota} {container_name}'
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_latency_jaeger():
    """Gets the latency from Jaeger using a specific script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir, '..')
    os.chdir(relative_path)
    cmd = ["python3", "jaegergrpc_service_avg.py", "grpc", "1", "socialnetwork-nginx-thrift-1"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract total_average_duration from the output
    match = re.search(r"Total average duration across all services: (\d+\.\d+) microseconds", result.stdout)
    if match:
        total_average_duration = float(match.group(1))
        rounded_duration = round(total_average_duration)
        return rounded_duration
    else:
        print("Total average duration not found in the script output.")
        return 0

def objective(x):
    """Objective function to minimize the L1 norm of x."""
    return np.sum(np.abs(x))

def constraints(x, A, y, eta):
    """Constraints ensuring that the Euclidean norm of the residuals is within eta."""
    return eta - np.linalg.norm(np.dot(A.T, x) - y)

def sigmoid_adjustment(values, scale=5):
    """Apply a sigmoid function to adjust CPU allocations within a soft limit, avoiding extreme adjustments for low values."""
    return 1 / (1 + np.exp(-scale * (values - 0.5)))

def optimize_resources(container_names, initial_cpu_allocations, iterations=20):
    """Optimize resource allocation for the given containers."""
    allocation_list = []
    cost_list = []
    time_stamps = []
    cpu_allocations = np.array(initial_cpu_allocations[:num_of_containers])
    np.random.seed(48)
    measurement_matrix = np.random.uniform(low=-1.0, high=1.0, size=(num_of_containers, num_of_measurements))
    gradient_history = []
    start_time = time.time()  # Start the timer

    for iteration in range(iterations):
        print("------ITERATION--------{}----------------".format(iteration))
        cost_neutral = [measure_cost(cpu_allocations)] * num_of_measurements
        sobol_generator = qmc.Sobol(d=num_of_measurements, scramble=True)
        random_vector = sobol_generator.random_base2(m=int(2**np.ceil(np.log2(num_of_measurements))))
        random_vector = random_vector[:num_of_measurements] * 0.2
        #print("shape of random_vector is {}".format(random_vector.shape))
        perturbation = np.dot(measurement_matrix, random_vector)
        delta = 0.2 / np.max(perturbation)
        perturbation = perturbation * delta
        cpu_allocations_reshaped = cpu_allocations.reshape(-1, 1)
        perturbed_allocations = cpu_allocations_reshaped + perturbation
        perturbed_allocations = np.clip(perturbed_allocations, 0.1, 1.0)  # Ensuring allocations stay within bounds
        #print("pertrubed allocation after clipping {}".format(perturbed_allocations))

        cost_measurements = []
        perturbed_allocations_transposed = np.transpose(perturbed_allocations)
        for alloc_set in perturbed_allocations_transposed:
            cost = measure_cost(alloc_set)
            #print("for allocation {} the cost is {}".format(alloc_set, cost))
            cost_measurements.append(cost)

        new_list = [(cost_measurements[i] - cost_neutral[i]) / np.squeeze(random_vector[i]) for i in range(num_of_measurements)]
        sum_list = [sum(x) for x in zip(*new_list)]
        observations = [x / num_of_measurements for x in sum_list]
        #print("---observations {}----".format(observations))

        eta = 0.2 / (1 + iteration * 0.05)
        cons = {'type': 'ineq', 'fun': constraints, 'args': (measurement_matrix, observations, eta)}
        result = minimize(objective, np.zeros(num_of_containers), method='SLSQP', constraints=[cons])
        estimated_gradient = result.x
        #print("gradient estimate before scaling -- {}".format(estimated_gradient))
        estimated_gradient = estimated_gradient / np.max(np.abs(estimated_gradient))
        #print("gradient estimate after scaling -- {}".format(estimated_gradient))

        learning_rate = 0.6 / (1 + iteration * 0.5)
        cpu_allocations -= learning_rate * estimated_gradient
        cpu_allocations = sigmoid_adjustment(cpu_allocations)  # Use sigmoid adjustment

        gradient_history.append(estimated_gradient)
        cost = measure_cost(cpu_allocations)
        #print("unregulated cost at iteration {}  is -- {}".format(iteration, cost))
        cost_list.append(cost)
        allocation_list.append(str(cpu_allocations))
        current_time = time.time() - start_time
        time_stamps.append(current_time)

    return cpu_allocations, cost_list, allocation_list, gradient_history, time_stamps

def measure_cost(cpu_allocations):
    """Measure the cost for given CPU allocations."""
    for index in range(len(container_names)):
        update_container_resources(container_names[index], cpu_allocations[index], -1)
    time.sleep(25)
    latency = get_latency_jaeger()
    #print("---latency---{}".format(latency))
    cost = latency_weight * (latency / 1000) + cpu_allocation_weight * (sum(cpu_allocations) / len(cpu_allocations))
    return cost

def plot_latency_vs_iteration(costs):
    """Plot the cost versus iteration."""
    plt.figure(figsize=(10, 6))
    plt.plot(costs, marker='', linestyle='-', color='b')  # No marker for the line
    plt.title("Cost vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    filename = "CONGO_costs_vs_iterations.png"
    plt.savefig("/your_placeholder/path_to_save_csv/{}".format(filename))  # Placeholder for actual path
    plt.show()

def store_convergence_data(allocation_list, costs, gradient_history, time_stamps):
    """Store convergence data in a CSV file."""
    data = pd.DataFrame({
        'cpu allocation': allocation_list,
        'cost observed': costs,
        'gradients': gradient_history,
        'time_stamps': time_stamps
    })
    filename = "CONGO_data.csv"
    filepath = "/your_placeholder/path_to_save_plot/{}".format(filename)  # Placeholder for actual path
    data.to_csv(filepath, index=False)


#--------------------------------
"""Running this code will generate a plot (costs vs iterations) and csv files"""
#-------------------------------
def main():

    initial_cpu_allocations = [0.7] * 10  # Starting point for CPU allocations

    # Optimize resources
    optimized_cpu_allocations, cost_list, allocation_list, gradient_history, time_stamps = optimize_resources(container_names, initial_cpu_allocations)

    # Plot and store results
    plot_latency_vs_iteration(cost_list)
    store_convergence_data(allocation_list, cost_list, gradient_history, time_stamps)

    # Stop all Docker containers
    hardcoded_password = "your_placeholder_password"  # Placeholder for actual password
    cmd = f'echo "{hardcoded_password}" | dzdo docker stop $(dzdo docker ps -a -q)'
    print(f"NOW EXECUTING {cmd}")
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
