### UNUSED ###

from microservice_architecture_simulator.registry import register_reward
import numpy as np

@register_reward("default")
def default_reward(param1, param2):
    return 0


@register_reward("negative_sum")
def negative_sum(latencies, resource_caps, lambda_):
    return -(sum(latencies) + lambda_ * np.sum(resource_caps))
