from templateRL import Agent
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange
import numpy as np

model = nn.Sequential(
    nn.Linear(8, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 4),
)

class RolloutData(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
writer = SummaryWriter()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

fit_step=0
def fit(batch):
    optim.zero_grad()
    o, a, r = batch
    o = o.float()
    r = r.float()
    y = torch.stack([pred[action] for pred, action in zip(model(o), a)])
    loss = F.mse_loss(y, r)
    loss.backward()
    optim.step()
    global fit_step
    writer.add_scalar('Loss/fit_step', loss, fit_step)
    fit_step+=1
    return loss
    

env = gym.make("LunarLander-v3")

class RolloutAgent(Agent):
    def act(self, observation):
        if np.random.rand() < 0.2: return env.action_space.sample()
        t = torch.from_numpy(observation)
        return model(t.unsqueeze(0)).argmax().item()

agent = RolloutAgent()
print("Starting training")
rounds = 50
episodes_per_round = 50
gamma=0.9
try:
    for i in trange(rounds): # 100 updates
        data = []
        model.eval()
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
        model.train()
        dl = DataLoader(RolloutData(data), 32, True, num_workers=11)
        losses = [fit(batch).detach() for batch in dl]
        writer.add_scalar('Loss/averaged', np.mean(losses), fit_step)


except KeyboardInterrupt:
    pass

env.close()

env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()
agent.play_episode(env, obs)
env.close()