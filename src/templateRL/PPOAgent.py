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
from typing import (
    Any,
    Callable,
    Literal
)
from .utils import deprecated
import time

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
    modules:list[Any] = [nn.Linear(*next(layers))]
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
                 hidden_layer_sizes:list[int],
                ):
        super().__init__()
        self._observation_size = observation_size
        self._action_size = action_size
        self._hidden_layer_sizes = hidden_layer_sizes
        self.value_net = make_value_net(observation_size, hidden_layer_sizes)
        self.policy_net = make_policy_net(observation_size, action_size, hidden_layer_sizes)
        self.value_net.eval()
        self.policy_net.eval()
        self.update_count = 0
        self.device = torch.device("cpu")
    
    @property
    def device(self):
        return self._device
    @device.setter
    def device(self, value):
        self._device = value
        self.policy_net.to(self._device)
        self.value_net.to (self._device)
    
    def _get_state(self):
        return {
            "value net state": self.value_net.state_dict(),
            "policy net state": self.policy_net.state_dict(),
            "observation size": self._observation_size,
            "action size": self._action_size,
            "hidden layer sizes": self._hidden_layer_sizes,
            "update count": self.update_count
        }
    @classmethod
    def _from_state(cls, state: dict):
        agent = PPOAgent(state['observation size'], state['action size'], state['hidden layer sizes'])
        agent.value_net.load_state_dict(state['value net state'])
        agent.policy_net.load_state_dict(state['policy net state'])
        agent.update_count = state['update count']
        return agent
    
    def save(self, f):
        torch.save(self._get_state(), f)
    @classmethod
    def load(cls, f) -> "PPOAgent":
        state = torch.load(f, weights_only=True)
        return cls._from_state(state)
    
    def act(self, observation):
        t = torch.from_numpy(observation).to(self.device)
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
        batch_size: int = 32,
        prog_bar: bool = False,
        log_mlflow: bool = False,
        rollout_collection_method: Literal['sequential', 'multi-process', 'multi-threaded'] = 'multi-process',
        timeout_minutes: int | None = None
    ):
        mlflow.log_params({
            "rounds": rounds,
            "episodes_per_round": episodes_per_round,
            "gamma": gamma,
            "epsilon": epsilon,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "rollout collection": rollout_collection_method
        }) if log_mlflow else None
        optim = torch.optim.Adam(chain(self.value_net.parameters(), self.policy_net.parameters()), learning_rate)
        round_iterator = trange if prog_bar else range
        match rollout_collection_method:
            case "sequential": collect_episodes = self.play_n_episodes_sequential
            case "multi-threaded": collect_episodes = self.play_n_episodes_in_thread_pool
            case "multi-process": collect_episodes = self.play_n_episodes_in_process_pool
            case _: raise ValueError(f"Unrecognized rollout_collection_method: {rollout_collection_method}")
        rollout_collection_time = 0
        model_uptate_time = 0
        start_time = time.monotonic()
        for training_round in round_iterator(rounds): # 100 updates
            if timeout_minutes is not None and time.monotonic()-start_time > timeout_minutes*60:
                break
            data = []
            self.policy_net.eval()
            self.value_net.eval()
            s = time.monotonic()
            rollouts = collect_episodes(env_factory, episodes_per_round, show_prog=False)
            t = time.monotonic()
            rollout_collection_time += (t-s)
            for j, rollout in enumerate(rollouts):
                obs, acts, rwds = rollout
                mlflow.log_metric("episode score", sum(rwds), training_round*episodes_per_round+j) if log_mlflow else None
                cum_rwds = [rwds.pop()]
                while rwds:
                    cum_rwds.append(gamma*cum_rwds[-1] + rwds.pop())  
                cum_rwds = reversed(cum_rwds)
                data.extend(zip(obs, acts, cum_rwds))

            self.policy_net.train()
            self.value_net.train()
            old_policy_net = copy_network(self.policy_net)
            old_policy_net.eval()
            dl = DataLoader(RolloutData(data), batch_size, True)
            losses = []
            s = time.monotonic()
            for batch in dl:
                optim.zero_grad()
                batch = [i.to(self.device) for i in batch]
                loss = batch_loss(batch, old_policy_net, epsilon, self.policy_net, self.value_net)
                loss.backward()
                optim.step()
                losses.append(loss.detach())
                mlflow.log_metric("batch loss", losses[-1], self.update_count) if log_mlflow else None
                self.update_count+=1
            t = time.monotonic()
            model_uptate_time += (t-s)

            
            mlflow.log_metric('training round loss', np.mean(losses).astype(float), training_round) if log_mlflow else None
            mlflow.log_metric("rollout collection time", rollout_collection_time)
            mlflow.log_metric("model update time", model_uptate_time)

    
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
            
        
        


if __name__=='__main__':
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
