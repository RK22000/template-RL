from templateRL import Agent
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange
import numpy as np
from itertools import chain

value_net = nn.Sequential(
    nn.Linear(8, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
)

policy_net = nn.Sequential(
    nn.Linear(8, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 4),
    nn.Softmax()
)

optim = torch.optim.Adam(chain(value_net.parameters(), policy_net.parameters()), 3e-4)

class RolloutData(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    
def advantage(observations, cumilative_rewards):
    """This is a measure of how much the actual cumilative rewards are better than the expected cumilative rewards"""
    return cumilative_rewards - value_net(observations)

def policy_ratio_and_entropy(observations, old_policy, actions):
    """Ratio of the likelyhood of doing an action with the policy against the older policy"""

    action_probs = policy_net(observations)
    distribution = Categorical(action_probs)
    log_prob = distribution.log_prob(actions)

    old_action_probs = old_policy(observations)
    old_distribution = Categorical(old_action_probs)
    old_log_prob = old_distribution.log_prob(actions)
    
    ratios = torch.exp(log_prob - old_log_prob)
    return ratios, old_distribution.entropy()

def clip_loss(ratio, advantages, epsilon):
    cpi_loss = ratio * advantages
    clipped_ratio_loss = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
    loss = torch.min(cpi_loss, clipped_ratio_loss)
    return loss

writer = SummaryWriter()
def fit_batch(batch, old_policy, epsilon):
    optim.zero_grad()
    o, a, r = batch
    o = o.float()
    r = r.float()

    advantages = advantage(o, r)
    ratio, entropy = policy_ratio_and_entropy(o, old_policy, a)
    clip = clip_loss(ratio, advantages, epsilon)
    
    policy_loss = -torch.mean(clip)
    
    value_loss = F.mse_loss(
        value_net(o),
        r
    )

    loss = policy_loss + 0.5*value_loss - 0.01*entropy
    
    loss.backward()
    optim.step()
    global fit_step
    writer.add_scalar('Loss/fit_step', loss, fit_step)
    fit_step+=1
    return loss

env = gym.make("LunarLander-v3")

class PPOAgent(Agent):
    def act(self, observation):
        t = torch.from_numpy(observation)
        action_probs = Categorical(policy_net(t))
        return action_probs.sample((1,))

agent = PPOAgent()
print("Starting training")
rounds = 50
episodes_per_round = 50
gamma=0.99
epsilon=0.2
try:
    for i in trange(rounds): # 100 updates
        data = []
        policy_net.eval()
        value_net.eval()
        for j in range(episodes_per_round): # 100 episodes for each update
            observation, _ = env.reset()
            obs, acts, rwds = agent.play_episode(env, observation)
            writer.add_scalar('Reward/episode_total', sum(rwds), i*episodes_per_round+j)
            running_total = 0
            cum_rwds = [rwds.pop()]
            while rwds:
                cum_rwds.append(gamma*cum_rwds[-1] + rwds.pop())  
            cum_rwds = reversed(cum_rwds)
            data.extend(zip(obs, acts, cum_rwds))
        policy_net.train()
        value_net.train()
        old_policy = policy_net.deep_copy()
        old_policy.eval()
        dl = DataLoader(RolloutData(data), 32, True, num_workers=11)
        losses = [fit_batch(batch, old_policy, epsilon).detach() for batch in dl]
        writer.add_scalar('Loss/averaged', np.mean(losses), fit_step)


except KeyboardInterrupt:
    pass

env.close()

env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()
agent.play_episode(env, obs)
env.close()