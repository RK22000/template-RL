from templateRL import PPOAgent
from templateRL.utils import mlflow_log_rollouts
import argparse
import gymnasium as gym
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument('-f', default='ppo.agent', help="Previously trained PPO agent file")
args = parser.parse_args()

factory = lambda: gym.make("LunarLander-v3")
def make_new_trained_ppo_agent(save_file: str):
    agent = PPOAgent(8, 4, [100,100])
    with mlflow.start_run(run_name='train ppo agent', nested=True) as run:
        agent.train(factory, 100, 50, log_mlflow=True, prog_bar=True)
        agent.save(save_file)

mlflow.set_experiment('save load demo')
with mlflow.start_run():
    try:
        agent = PPOAgent.load(args.f)
    except FileNotFoundError:
        print(f'file {args.f} not found')
        print('Training new agent')
        make_new_trained_ppo_agent(args.f)
        agent = PPOAgent.load(args.f)
    print("Evaluating agent")
    mlflow.log_artifact(args.f)
    rollouts = agent.play_n_episodes_parallel_processed(factory, 10, show_prog=True)
    mlflow_log_rollouts(rollouts, {'agent path': args.f})





    