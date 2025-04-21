"""
Train agent on specific hyper parameters. Find the best hyper parameters for training.
Train and save a model.
"""
from templateRL import PPOAgent
from templateRL.utils import mlflow_log_rollouts
import argparse
import gymnasium as gym
import time
import mlflow
import numpy as np
import json
from itertools import product

parser = argparse.ArgumentParser()
# parser.add_argument('-f', default='ppo.agent', help="Previously trained PPO agent file")
args = parser.parse_args()

EXPERIMENT_NAME = "Train PPO agent"
agentfilename = 'ppo.agent'
timeout_on_one_runs = 20
factory = lambda: gym.make("LunarLander-v3")
new_agent = lambda: PPOAgent(8, 4, [100,100])

def run_experiment(rounds, episodes_per_round, batch_size, eval_episodes):
    agent = new_agent()
    agent.train(
        env_factory=factory,
        rounds=rounds,
        episodes_per_round=episodes_per_round,
        batch_size=batch_size,
        log_mlflow=True,
        rollout_collection_method='multi-process',
        timeout_minutes=timeout_on_one_runs
    )
    agent.save(agentfilename)
    mlflow.log_artifact(agentfilename)
    s = time.monotonic()
    rollouts = agent.play_n_episodes_in_process_pool(factory, eval_episodes)
    t = time.monotonic()
    mlflow.log_metric("model evaluation time", t-s)
    mlflow.log_metric("averages episode score", np.mean([sum(i.rewards) for i in rollouts]))

hyperparams = {
    "rounds": [250, 150, 50],
    "episodes_per_round": [250, 150, 50],
    "batch_size": [512, 128, 32]
}

# Get the keys and the list of values
keys = hyperparams.keys()
values = hyperparams.values()

mlflow.set_experiment(EXPERIMENT_NAME)
# Iterate through all combinations
for combo in product(*values):
    param_set = dict(zip(keys, combo))
    print("Doing experiment with")
    print(json.dumps(param_set, indent=2))
    with mlflow.start_run():
        run_experiment(eval_episodes=100, **param_set)





# def train(agent: PPOAgent, params: dict):
    

# def make_new_trained_ppo_agent(save_file: str):
#     agent = PPOAgent(8, 4, [100,100])
#     # if torch.cuda.is_available():
#     #     agent.device = torch.device("cuda")
#     with mlflow.start_run(run_name='train ppo agent', nested=True) as run:
#         agent.train(factory, 100, 50, log_mlflow=True, prog_bar=True)
#         agent.save(save_file)

# mlflow.set_experiment('save load demo')
# with mlflow.start_run():
#     try:
#         agent = PPOAgent.load(args.f)
#         # if torch.cuda.is_available():
#         #     print("Useing cuda")
#         #     agent.device = torch.device("cuda")
#     except FileNotFoundError:
#         print(f'file {args.f} not found')
#         print('Training new agent')
#         make_new_trained_ppo_agent(args.f)
#         agent = PPOAgent.load(args.f)
#     print("Evaluating agent")
#     mlflow.log_artifact(args.f)
#     # rollouts = agent.play_n_episodes_parallel_processed(factory, 10, show_prog=True)
#     rollouts = agent.play_n_episodes_sequential(factory, 10, show_prog=True)
#     mlflow_log_rollouts(rollouts, {'agent path': args.f})





    

