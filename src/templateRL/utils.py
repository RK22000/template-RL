import mlflow
from collections import namedtuple
import numpy as np
from functools import wraps
import warnings
from typing import Callable

Rollout = namedtuple('Rollout', ['observations', 'actions', 'rewards'])

def rewards_to_cumilative_rewards(rewards: list[float], gamma: float):
    cum_rewards = [rewards.pop()]
    while rewards:
        cum_rewards.append(gamma*cum_rewards[-1] + rewards.pop())  
    cum_rewards = reversed(cum_rewards)
    return cum_rewards

def mlflow_log_rollouts(rollouts: list[Rollout], params: dict = {}):
    """Log each rollout as a child run with params"""
    for i, rollout in enumerate(rollouts):
        _, _, rewards = rollout
        with mlflow.start_run(run_name=f"Rollout {i:0=5}", nested=True) as run:
            mlflow.log_params(params)
            for j, score in enumerate(np.cumsum(rewards)):
                mlflow.log_metric('score', score, j)

def deprecated(reason: str):
    def decorator(func):
        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(f'function {func.__qualname__} is deprecated. {reason}', stacklevel=2)
            return func(*args, **kwargs)
        return deprecated_func
    return decorator
    