from .Agent import Agent
import numpy as np

class RandomAgent(Agent):
    def __init__(self, action_size:int, seed=42):
        super().__init__()
        self.action_size = action_size
        self.rng = np.random.default_rng(seed)
    def act(self, observation):
        return self.rng.integers(self.action_size)
        