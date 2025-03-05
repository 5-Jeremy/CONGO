from ray.rllib.models import ModelCatalog

ENVS = {}
REWARDS = {}
WORKLOADS = {}
LATENCY_FUNCTIONS = {}

# Note that some of these decorators are not used in the current implementation

def register_env(name: str):
    """Decorator for registering an env."""

    def decorator(cls):
        ENVS[name] = cls
        return cls

    return decorator


def register_reward(name: str):
    """Decorator for registering a reward."""

    def decorator(func):
        REWARDS[name] = func
        return func

    return decorator

def register_latency_function(name: str):
    """Decorator for registering a reward."""

    def decorator(func):
        LATENCY_FUNCTIONS[name] = func
        return func

    return decorator

def register_workload(name: str):
    """Decorator for registering a workload generating function."""

    def decorator(func):
        WORKLOADS[name] = func
        return func

    return decorator

def register_model(name: str):
    """Decorator for registering a model."""

    def decorator(cls):
        ModelCatalog.register_custom_model(name, cls)
        return cls

    return decorator


def create_reward(name):
    return REWARDS[name]

def create_latency_function(name):
    return LATENCY_FUNCTIONS[name]

def create_workload(name):
    return WORKLOADS[name]