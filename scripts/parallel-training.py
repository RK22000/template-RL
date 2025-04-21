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
import mlflow


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
    nn.Softmax(-1)
)

old_policy_net = nn.Sequential(
    nn.Linear(8, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 4),
    nn.Softmax(-1)
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
batch_num=0
def fit_batch(batch, old_policy, epsilon):
    optim.zero_grad()
    o, a, r = batch
    o = o.float()
    r = r.float()

    advantages = advantage(o, r)
    ratio, entropy = policy_ratio_and_entropy(o, old_policy, a)
    clip = clip_loss(ratio, advantages, epsilon)
    
    policy_loss = -torch.mean(clip)
    
    v = value_net(o).reshape(r.shape)
    value_loss = F.mse_loss(
        v,
        r
    )

    loss = torch.mean(policy_loss + 0.5*value_loss - 0.01*entropy)
    loss.backward()
    optim.step()
    global batch_num
    writer.add_scalar('Loss/batch loss', loss, batch_num)
    mlflow.log_metric("batch loss", loss, batch_num)
    batch_num+=1
    return loss

env_factory = lambda: gym.make("LunarLander-v3")

class PPOAgent(Agent):
    def act(self, observation):
        t = torch.from_numpy(observation)
        p = policy_net(t)
        action_probs = Categorical(p)
        return action_probs.sample().item()

agent = PPOAgent()
print("Starting training")
rounds = 50
episodes_per_round = 50
gamma=0.99
epsilon=0.2
mlflow.log_params({
    "Agent": 'PPOAgent-parallel-process',
    "rounds": rounds,
    "episodes_per_round": episodes_per_round,
    "gamma": gamma,
    "epsilon": epsilon
})
try:
    for i in trange(rounds): # 100 updates
        data = []
        policy_net.eval()
        value_net.eval()
        rollouts = agent.play_n_episodes_in_process_pool(env_factory, episodes_per_round)
        for j, rollout in enumerate(rollouts):
            obs, acts, rwds = rollout
            writer.add_scalar('Reward/epsiode reward', sum(rwds), i*episodes_per_round+j)
            mlflow.log_metric("episode score", sum(rwds), i*episodes_per_round+j)
            running_total = 0
            cum_rwds = [rwds.pop()]
            while rwds:
                cum_rwds.append(gamma*cum_rwds[-1] + rwds.pop())  
            cum_rwds = reversed(cum_rwds)
            data.extend(zip(obs, acts, cum_rwds))

        policy_net.train()
        value_net.train()
        old_policy_net.load_state_dict(policy_net.state_dict(),)
        old_policy_net.eval()
        dl = DataLoader(RolloutData(data), 32, True, num_workers=11)
        losses = [fit_batch(batch, old_policy_net, epsilon).detach() for batch in dl]
        writer.add_scalar('Loss/epoch loss', np.mean(losses), batch_num)
        mlflow.log_metric('epoch loss', np.mean(losses), batch_num)



except KeyboardInterrupt:
    pass


env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()
_, _, rewards = agent.play_episode(env, obs)
env.close()
for i, s in enumerate(np.cumsum(rewards)):
    writer.add_scalar("Reward/demo-episode-socre", s, i)
    mlflow.log_metric("demo episode score", s, i)