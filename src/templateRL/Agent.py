from abc import ABC, abstractmethod
import gymnasium as gym
from typing import (
    Any
)


""" 
Some asumptions
Agents will always run for an episode while training

"""
class Agent(ABC):
    @abstractmethod
    def act(self, observation) -> Any:
        NotImplementedError()
    
    def play_episode(self, env: gym.Env, initial_observation: Any):
        """
        Play an episode in the given gymnasium environment
        The environment will liekly need to be reset before calling this method

        Returns
        -------

            a 3-tuple of observations, action, and rewards
        """
        observation=initial_observation
        episode_over = False
        observations, actions, rewards = [], [], []
        while not episode_over:
            action = self.act(observation)
            observations.append(observation)
            actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            episode_over = terminated or truncated
        return (observations, actions, rewards)