import random
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import os
import pandas as pd
import yaml
import sys
import subprocess
from scipy.stats import qmc

# Append paths for importing required modules
sys.path.append('../cosamp_package')
from Cosamp import cosamp

# Import the ZORO utility (ensure that this is correctly imported)
sys.path.append('../')
from ZORO_utils.optimizers import ZORO
import pyproximal as pyprox

#============================
# GLOBAL VARIABLES DECLARATION
#============================
latency_weight = 1.0  # Weight for latency in cost calculation
cpu_allocation_weight = 10.0  # Weight for CPU allocation in cost calculation

# List of container names to be optimized
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

# Total number of containers in the social network application
num_of_containers = len(container_names)

#============================
# FUNCTION DEFINITIONS
#============================

def update_container_resources(container_name, cpu_allocation, memory_allocation):
    """
    Update the CPU allocation for a given container.

    Parameters:
    container_name (str): Name of the container.
    cpu_allocation (float): CPU allocation for the container.
    memory_allocation (int): Memory allocation in MB. Set to -1 to skip memory update.
    """
    cpu_period = 100000
    cpu_quota = int(cpu_allocation * cpu_period)
    hardcoded_password = "your_placeholder_password"  # Placeholder for actual password

    if memory_allocation != -1:
        cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpus={cpu_allocation} --memory={memory_allocation}m --memory-swap={memory_allocation}m {container_name}'
    else:
        cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpu-period={cpu_period} --cpu-quota={cpu_quota} {container_name}'

    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_latency_jaeger():
    """
    Retrieves the system latency using Jaeger through a script.

    Returns:
    int: Rounded average latency from Jaeger in microseconds.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir, '..')
    os.chdir(relative_path)

    cmd = ["python3", "jaegergrpc_service_avg.py", "grpc", "1", "socialnetwork-nginx-thrift-1"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract total average duration from the output
    match = re.search(r"Total average duration across all services: (\d+\.\d+) microseconds", result.stdout)
    if match:
        total_average_duration = float(match.group(1))
        rounded_duration = round(total_average_duration)
        return rounded_duration
    else:
        print("Total average duration not found in the script output.")
        return 0


def measure_cost(cpu_allocations):
    """
    Measure system cost based on CPU allocations and system latency.

    Parameters:
    cpu_allocations (list): List of CPU allocations for containers.

    Returns:
    float: Calculated cost based on latency and CPU allocation.
    """
    for index in range(0, len(container_names)):
        update_container_resources(container_names[index], cpu_allocations[index], -1)

    time.sleep(20)  # Wait for the system to stabilize
    latency = get_latency_jaeger()
    print(f"---Latency: {latency}---")
    cost = latency_weight * (latency / 1000) + cpu_allocation_weight * (sum(cpu_allocations) / len(cpu_allocations))
    return cost


def eval_cost(x):
    """
    Wrapper function for cost evaluation.

    Parameters:
    x (list): Current CPU allocations.

    Returns:
    int: System latency from Jaeger.
    """
    return get_latency_jaeger()  # Could also call measure_cost(x) for detailed cost


def optimize_resources(container_names, initial_cpu_allocations, iterations=15, num_of_measurements=13, delta=0.1, s=7):
    """
    Perform resource optimization using the ZORO algorithm.

    Parameters:
    container_names (list): List of container names to optimize.
    initial_cpu_allocations (list): Initial CPU allocation for containers.
    iterations (int): Number of optimization iterations.
    num_of_measurements (int): Number of measurements for gradient estimation.
    delta (float): Perturbation parameter for ZORO.
    s (int): Sparsity parameter for ZORO.

    Returns:
    tuple: Optimized CPU allocations, cost list, allocation list, timestamps list.
    """
    allocation_list = []
    cost_list = []
    cpu_allocations = np.array(initial_cpu_allocations[:num_of_containers])
    timestamps_list = []

    # Initial cost at starting allocation
    cost = measure_cost(cpu_allocations)
    cost_list.append(cost)
    allocation_list.append(str(cpu_allocations))

    # Initialize measurement matrix A
    m = num_of_measurements
    A = np.random.normal(0, 1, (m, num_of_containers))  # Measurement matrix

    iteration_start_time = time.time()
    timestamps_list.append(time.time() - iteration_start_time)

    # Initialize ZORO optimizer
    lr = 0.15  # Learning rate
    prox = pyprox.Box(lower=np.array([0.05]*num_of_containers), upper=np.array([0.99]*num_of_containers))  # Resource bounds
    zoro_params = {"delta": delta, "sparsity": s, "step_size": lr, "num_samples": m}
    zoro = ZORO(cpu_allocations, eval_cost, zoro_params, prox=prox)

    # Optimization loop
    for steps in range(iterations):
        print(f"------Iteration {steps}-------------")

        # Measure current cost
        cost = measure_cost(zoro.x)
        cost_list.append(cost)

        # Perform ZORO optimization step
        zoro.step()

        # Clip CPU allocations to be within defined bounds
        zoro.x = np.clip(zoro.x, 0.05, 0.99)

        # Measure cost after optimization step
        cost = measure_cost(zoro.x)
        print(f"Cost at iteration {steps}: {cost}")

        # Record iteration time
        iteration_end_time = time.time()
        iteration_time = iteration_end_time - iteration_start_time
        timestamps_list.append(iteration_time)

        print(f"Time taken for iteration {steps}: {iteration_time:.6f} seconds")
        allocation_list.append(str(zoro.x))

        time.sleep(5)  # Allow some time for the changes to take effect

    return zoro.x, cost_list, allocation_list, timestamps_list


def plot_cost_vs_iteration(costs, initial_cpu_start_value):
    """
    Plot cost vs iteration and save the plot.

    Parameters:
    costs (list): List of costs at each iteration.
    initial_cpu_start_value (float): Initial CPU start value for filename.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs, linestyle='-', color='b')
    plt.title("Cost vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    filename = f"ZORO_cost_vs_iteration_{initial_cpu_start_value}_fixed_type_variable_workload_1.png"
    plt.savefig(f"/your_placeholder/path_to_save_plot/{filename}")
    plt.show()


# Plot cost vs time
def plot_cost_vs_time(costs, times, initial_cpu_start_value):
    """
    Plot cost vs time and save the plot.

    Parameters:
    costs (list): List of costs.
    times (list): List of time taken for each iteration.
    initial_cpu_start_value (float): Initial CPU start value for filename.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(times, costs, marker='', linestyle='-', color='g')
    plt.title("Cost vs. Time Taken Per Iteration")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Cost")
    plt.grid(True)
    filename = f"ZORO_cost_vs_time_{initial_cpu_start_value}_fixed_type_variable_workload_1.png"
    plt.savefig(f"/your_placeholder/path_to_save_plot/{filename}")
    plt.show()

    Per Iteration")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Cost")
    plt.grid(True)
    filename = f"ZORO_cost_vs_time_{initial_cpu_start_value}_fixed_type_variable_workload_1.png"
    plt.savefig(f"/your_placeholder/path_to_save_plot/{filename}")
    plt.show()


def store_convergence_data(iterations, allocation_list, costs, times, initial_cpu_start_value):
    """
    Store the convergence data in a CSV file.

    Parameters:
    iterations (range): Range of iterations.
    allocation_list (list): List of CPU allocations per iteration.
    costs (list): List of costs observed per iteration.
    times (list): List of timestamps per iteration.
    initial_cpu_start_value (float): Initial CPU allocation value for filename.
    """
    data = pd.DataFrame({
        'Iterations': iterations,
        'CPU Allocation': allocation_list,
        'Cost Observed': costs,
        'timestamps': times
    })
    filename = f"ZORO_workload_{initial_cpu_start_value}_fixed_type_variable_workload_1.csv"
    filepath = f"/your_placeholder/path_to_save_plot/{filename}"
    data.to_csv(filepath, index=False)


#============================
# MAIN DRIVER FUNCTION
#============================

def optimize_all_containers():
    """
    Main function to optimize CPU allocations for all containers.
    """
    initial_cpu = 0.35  # Starting point for CPU allocations
    initial_cpu_allocations = [initial_cpu] * num_of_containers
    num_of_measurements = 10
    max_iterations = 13

    # Perform the optimization
    optimized_cpu_allocations, cost_list, allocation_list, timestamps_list = optimize_resources(
        container_names, initial_cpu_allocations, max_iterations, num_of_measurements
    )

    # Store and plot the results
    store_convergence_data(range(max_iterations + 1), allocation_list, cost_list, timestamps_list, initial_cpu)
    plot_cost_vs_iteration(cost_list, initial_cpu)
    plot_cost_vs_time(cost_list, timestamps_list, initial_cpu)


#============================
# ENTRY POINT
#============================

if __name__ == "__main__":
    optimize_all_containers()
