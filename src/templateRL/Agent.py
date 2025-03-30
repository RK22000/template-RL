from abc import ABC, abstractmethod
import gymnasium as gym
from tqdm import trange, tqdm
from typing import (
    Any,
    Callable
)
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    ProcessPoolExecutor,
)


""" 
Some asumptions
Agents will always run for an episode while training
The base agent will be generic and as much as possible not coupled with
libraries like pytorch, tensorflow, sklearn or others
"""
class Agent(ABC):
    @abstractmethod
    def act(self, observation) -> Any:
        NotImplementedError()
    def __call__(self, *args, **kwds):
        return super().act(*args, **kwds)
        
    def __init__(self):
        self._decorators = []
        """
        Names of the decorators applied on the agent.
        This is likely for book keeping on the library's part.
        Users don't need to worry about this
        """
    
    # optional decorator to return cumilative rewards instead of regular rewards
    def play_episode(self, env: gym.Env, initial_observation: Any|None = None):
        """
        Play an episode in the given gymnasium environment
        If initial_observation is provided, the agent will continue in the environment 
        from that observation.
        Else it will reset the environment and start fresh

        Returns
        -------

            a 3-tuple of observations, action, and rewards
        """
        if initial_observation is None:
            initial_observation, _ = env.reset()
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
        if initial_observation is None:
            env.close()
        return (observations, actions, rewards)
    
    def play_n_episodes_sequential(
        self, 
        env_factory: Callable[[], gym.Env], 
        n:int, 
        show_prog:bool=False):
        """
        Play multiple episodes in an environment
        This method will reset the environment before each episode

        Args:
            env (gym.Env): _description_
            n (int): _description_
        """
        env = env_factory()
        rollouts = []
        r = trange if show_prog else range
        for _ in r(n):
            initial_obs, _ = env.reset()
            rollout = self.play_episode(env, initial_obs)
            rollouts.append(rollout)
        return rollouts
    
    def play_n_episodes_parallel_threaded(self, env_factory: Callable[[],gym.Env], n:int, n_workers:int=None, show_prog:bool=False):
        """multi threaded roll out collection

        Args:
            env_factory (Callable[[],gym.Env]): _description_
            n (int): _description_
            n_workers (int, optional): _description_. Defaults to None.
            show_prog (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        rollouts = []
        bar = iter(trange(n))
        def func():
            env = env_factory()
            rollout = self.play_episode(env)
            env.close()
            return rollout
        with ThreadPoolExecutor(n_workers) as executor:
            futures = []
            for _ in range(n):
                future = executor.submit(func)
                futures.append(future)
            for future in as_completed(futures):
                rollout = future.result()
                if show_prog: next(bar)
                rollouts.append(rollout)
        return rollouts

    def play_n_episodes_parallel_processed(self, env_factory: Callable[[],gym.Env], n:int, n_workers:int=None, show_prog:bool=False):
        """Play multiple episodes in parallel processes

        Args:
            env_factory (Callable[[],gym.Env]): _description_
            n (int): _description_
            n_workers (int, optional): _description_. Defaults to None.
            show_prog (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        rollouts = []
        if show_prog:
            bar = iter(trange(n))
        with ProcessPoolExecutor(n_workers) as executor:
            futures = []
            for _ in range(n):
                future = executor.submit(self.play_episode, env_factory())
                futures.append(future)
            for future in as_completed(futures):
                rollout = future.result()
                if show_prog: next(bar)
                rollouts.append(rollout)
        return rollouts

            


# class MultithreadedAgent(Agent):
#     """
#     This type of agent is designed for running in multiple environments using multi thread
#     """