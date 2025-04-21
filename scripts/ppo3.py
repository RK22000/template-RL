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
import io
from typing import Callable
from templateRL.utils import deprecated

def copy_network(network):
    buffer = io.BytesIO()
    torch.save(network, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)
    
class RolloutData(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    
def advantage(observations, cumilative_rewards, value_net):
    """This is a measure of how much the actual cumilative rewards are better than the expected cumilative rewards"""
    return cumilative_rewards - value_net(observations)

def policy_ratio_and_entropy(observations, old_policy, actions, policy_net):
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

def batch_loss(batch, old_policy, epsilon, policy_net, value_net):
    o, a, r = batch
    o = o.float()
    r = r.float()

    advantages = advantage(o, r, value_net)
    ratio, entropy = policy_ratio_and_entropy(o, old_policy, a, policy_net)
    clip = clip_loss(ratio, advantages, epsilon)
    
    policy_loss = -torch.mean(clip)
    
    v = value_net(o).reshape(r.shape)
    value_loss = F.mse_loss(
        v,
        r
    )

    loss = torch.mean(policy_loss + 0.5*value_loss - 0.01*entropy)
    return loss

def make_network(layer_sizes: list[int]):
    layers = iter((i,j) for i,j in zip(layer_sizes[:-1], layer_sizes[1:]))
    modules = [nn.Linear(*next(layers))]
    for l in layers:
        modules.append(nn.ReLU())
        modules.append(nn.Linear(*l))
    return nn.Sequential(*modules)

def make_value_net(observation_size: int, hidden_layer_sizes: list[int]):
    layer_sizes = [observation_size] + hidden_layer_sizes + [1]
    return make_network(layer_sizes)

def make_policy_net(observation_size: int, action_size: int, hidden_layer_sizes: list[int]):
    layer_sizes = [observation_size] + hidden_layer_sizes + [action_size]
    network = make_network(layer_sizes)
    network.append(nn.Softmax(-1))
    return network

class PPOAgent(Agent):
    def __init__(self, 
                 observation_size:int, 
                 action_size:int, 
                 hidden_layer_sizes:int):
        self.value_net = make_value_net(observation_size, hidden_layer_sizes)
        self.policy_net = make_policy_net(observation_size, action_size, hidden_layer_sizes)
        self.value_net.eval()
        self.policy_net.eval()
        self.update_count = 0
    
    def act(self, observation):
        t = torch.from_numpy(observation)
        p = self.policy_net(t)
        action_probs = Categorical(p)
        return action_probs.sample().item()
    def train(
        self,
        env_factory: Callable[[], gym.Env],
        rounds: int,
        episodes_per_round: int,
        gamma: float=0.99,
        epsilon: float=0.2,
        learning_rate: float=3e-4,
        prog_bar: bool = False,
        log_mlflow: bool = False
    ):
        mlflow.log_params({
            "Agent": 'PPOAgent-parallel-process',
            "rounds": rounds,
            "episodes_per_round": episodes_per_round,
            "gamma": gamma,
            "epsilon": epsilon
        }) if log_mlflow else None
        optim = torch.optim.Adam(chain(self.value_net.parameters(), self.policy_net.parameters()), learning_rate)
        round_iterator = trange if prog_bar else range
        for i in round_iterator(rounds): # 100 updates
            data = []
            self.policy_net.eval()
            self.value_net.eval()
            rollouts = self.play_n_episodes_in_process_pool(env_factory, episodes_per_round, show_prog=False)
            for j, rollout in enumerate(rollouts):
                obs, acts, rwds = rollout
                mlflow.log_metric("episode score", sum(rwds), i*episodes_per_round+j)
                running_total = 0
                cum_rwds = [rwds.pop()]
                while rwds:
                    cum_rwds.append(gamma*cum_rwds[-1] + rwds.pop())  
                cum_rwds = reversed(cum_rwds)
                data.extend(zip(obs, acts, cum_rwds))

            self.policy_net.train()
            self.value_net.train()
            old_policy_net = copy_network(self.policy_net)
            old_policy_net.eval()
            dl = DataLoader(RolloutData(data), 32, True, num_workers=11)
            losses = []
            for batch in dl:
                optim.zero_grad()
                loss = batch_loss(batch, old_policy_net, epsilon, self.policy_net, self.value_net)
                loss.backward()
                optim.step()
                losses.append(loss.detach())
                mlflow.log_metric("batch loss", losses[-1], self.update_count) if log_mlflow else None
                self.update_count+=1
            mlflow.log_metric('epoch loss', np.mean(losses), self.update_count) if log_mlflow else None

    # Todo make this general to all agents
    @deprecated("Get rollouts directly and call utils.mlflow_log_rollouts instead")
    def evaluate(
        self,
        env_factory: Callable[[], gym.Env],
        episodes: int = 50,
        prog_bar: bool = False,
        log_mlflow: bool = False
    ):
        rollouts = self.play_n_episodes_in_process_pool(env_factory, episodes, show_prog=prog_bar)
        for i, rollout in enumerate(rollouts):
            _, _, rewards = rollout
            with mlflow.start_run(run_name=f"Evaluation runs {i:0=5}", nested=True) as run:
                score = 0
                for j, r in enumerate(rewards):
                    score += r
                    mlflow.log_metric('score', score, j)
            
        
        


agent = PPOAgent(8, 4, [100,100])
print("Starting training")

factory = lambda: gym.make("LunarLander-v3")
agent.train(
    env_factory = factory,
    rounds=50,
    episodes_per_round=50,
    gamma=0.99,
    epsilon=0.2,
    prog_bar=True
)

agent.evaluate(
    env_factory=factory,
    prog_bar=True,
)



env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()
_, _, rewards = agent.play_episode(env, obs)
env.close()
