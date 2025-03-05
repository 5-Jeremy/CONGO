### UNUSED ###

from microservice_architecture_simulator.registry import register_latency_function

@register_latency_function("inverse_cap")
def inverse_cap_latency(resource_caps, latency_weights):
    return sum([x / (y + 2.5) for x, y in zip(latency_weights, resource_caps)])


@register_latency_function("constant_latency")
def dummy_latency():
    return 1
