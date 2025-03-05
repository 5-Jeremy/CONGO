import random
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import os
import pandas as pd

#============================
### GLOBAL VARIABLES DECLARATION
#============================
# Weights for cost calculation
latency_weight = 1.0
cpu_allocation_weight = 10.0


def update_container_resources(container_name, cpu_allocation, memory_allocation):
    """Update the CPU allocation for a given container."""
    cpu_period = 100000
    cpu_quota = int(cpu_allocation * cpu_period)
    hardcoded_password = "Poornima@021972"
    if memory_allocation != -1:
        cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpus={cpu_allocation} --memory={memory_allocation}m --memory-swap={memory_allocation}m {container_name}'
    else:
        cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpu-period={cpu_period} --cpu-quota={cpu_quota} {container_name}'  
        #cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpus={cpu_allocation} {container_name}'
    #print(f"NOW EXECUTING {cmd}")
    subprocess.run(cmd, shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_latency_jaeger():
    os.chdir("/home/grads/p/prathikvijaykumar/DeathStarBench/socialNetwork")
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



def measure_cost(cpu_allocations):
    """Measure the cost for given CPU allocations."""
    for index in range(len(container_names)):
        update_container_resources(container_names[index], cpu_allocations[index], -1)
    time.sleep(20)
    latency = get_latency_jaeger()
    cost = latency_weight * (latency / 1000) + cpu_allocation_weight * (sum(cpu_allocations) / len(cpu_allocations))
    return cost


def optimize_all_container_with_sgdsp(container_names, initial_cpu=0.5, initial_learning_rate=0.05, max_iterations=10, delta=0.02):
    global_cpu_allocations = {name: initial_cpu for name in container_names}
    global_latencies = []
    global_costs = []
    learning_rate = initial_learning_rate

    num_nodes = len(container_names)
    action = np.array([initial_cpu] * num_nodes, dtype=np.float32)
    steps = 0

    while steps < max_iterations:
        cost = measure_cost(action)
        grad_estimates = np.zeros((num_nodes,))
        for i in range(num_nodes):
            print("------{}-------------".format(i))
            U = np.sign(np.random.binomial(1, 0.5, size=num_nodes) - 0.5)
            perturbed_action = action + delta * U
            perturbed_cost = measure_cost(perturbed_action)
            grad_est = ((perturbed_cost - cost) / delta) * U
            grad_estimates += grad_est
        grad_dir_vector = grad_estimates / num_nodes
        grad_dir_vector = grad_dir_vector / np.linalg.norm(grad_dir_vector)
        
        change = grad_dir_vector * learning_rate
        action_next = action - change
        action = np.clip(action_next, 0.1, 1.0)
        
        
        global_costs.append(cost)
        
        steps += 1
        learning_rate *= (1.0 / (1.0 + steps))
    
    return global_costs


def plot_optimization_results(global_costs):
    plt.figure(figsize=(10, 6))
    plt.plot(global_costs, marker='o', linestyle='-', color='b')
    plt.title("Cost vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("cost = latency + cost of using resource ")
    filename = f"SGDSP_plot_variable_workload.png"
    plt.grid(True)
    plt.savefig(f"/home/grads/p/prathikvijaykumar/results_for_paper/vanilla/{filename}")  # Placeholder for actual path
    plt.show()

def store_optimization_data(iterations, global_costs):
    df = pd.DataFrame({
        'Iterations': iterations,
        'Cost': global_costs
    })
    filename = f"SGDSP_variable_workload.csv"
    filepath = f"/home/grads/p/prathikvijaykumar/results_for_paper/vanilla/{filename}"
    df.to_csv(filepath, index=False)

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


def optimize_all_containers():
    initial_cpu = 0.7
    initial_learning_rate = 0.2
    max_iterations = 20
    global_costs = optimize_all_container_with_sgdsp(container_names, initial_cpu, initial_learning_rate, max_iterations)
    plot_optimization_results(global_costs)
    store_optimization_data(range(max_iterations), global_costs)

optimize_all_containers()
hardcoded_password = "your_placeholder_password"
cmd = f'echo "{hardcoded_password}" | dzdo docker stop $(dzdo docker ps -a -q)'
subprocess.run(cmd, shell=True)
