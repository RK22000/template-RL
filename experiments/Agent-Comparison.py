"""
In this experiment we compare the training and performance of different agents 
on the moon landing gymnasium environment

We will run the experiment with the following agents:

    RandomAgent
    PPOAgent
"""

if __name__!='__main__':
    raise Exception(f"{__file__} should only be called as a script. It should not \
be imported.")

from templateRL import (
    RandomAgent,
    PPOAgent
)
import mlflow
from templateRL.utils import mlflow_log_rollouts
import gymnasium as gym

EXPERIMENT_NAME = "Agent Comparison"

factory = lambda: gym.make("LunarLander-v3")
observation_size=8
action_size = 4

def context_as_decorator(context):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with context():
                return func(*args, **kwargs)
        return wrapper
    return decorator

# @context_as_decorator(mlflow.start_run(run_name='random agent', nested=True))
def random_agent_experiment(
    evaluation_episodes: int
):
    random_agent = RandomAgent(action_size)
    mlflow.log_param("Agent", "RandomAgent")
    print("Collecting rollouts to evaluate random agent")
    rollouts = random_agent.play_n_episodes_parallel_processed(factory, evaluation_episodes, show_prog=True)
    mlflow_log_rollouts(rollouts)

# @context_as_decorator(mlflow.start_run(run_name='ppo agent', nested=True))
def ppo_agent_experiment(
    evaluation_episodes: int,
):
    ppo_agent = PPOAgent(observation_size, action_size, [100,100])
    mlflow.log_param("Agent", "PPOAgent")
    print('Training ppo agent')
    ppo_agent.train(
        env_factory=factory,
        rounds=50,
        episodes_per_round=50,
        prog_bar=True,
        log_mlflow=True
    )
    print("Collecting rollouts to evaluate ppo agent")
    rollouts = ppo_agent.play_n_episodes_parallel_processed(factory, evaluation_episodes, show_prog=True)
    mlflow_log_rollouts(rollouts)
    

mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run():
    evaluation_runs = 10
    with mlflow.start_run(run_name='random agent', nested=True):
        random_agent_experiment(evaluation_runs)
    with mlflow.start_run(run_name='ppo agent', nested=True):
        ppo_agent_experiment(evaluation_runs)

    

    


