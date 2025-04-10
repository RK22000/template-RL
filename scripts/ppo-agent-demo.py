from templateRL import PPOAgent
import gymnasium as gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', default='ppo.agent', help="Previously trained PPO agent file")
args = parser.parse_args()

env = gym.make("LunarLander-v3", render_mode='human')

agent = PPOAgent.load(args.f)
agent.play_episode(env)