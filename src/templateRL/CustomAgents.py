from .Agent import Agent
import numpy as np
from .RandomAgent import RandomAgent
from typing import Any

class CompoundAgent(Agent):
    """A composite agent that delegates actions to multiple sub-agents based on given weights.

    This agent randomly selects one of its sub-agents according to the provided probability
    weights and uses that agent to determine the action for the current observation.

    Args:
        agents (list[Agent]): List of agent instances to delegate actions to
        weights (list[float]): Probability weights for selecting each agent. Will be normalized
            to sum to 1. If fewer weights than agents are provided, remaining weights will be
            distributed evenly from the remaining probability mass.
    """
    def __init__(self, agents: list[Agent], weights: list[float]) -> None:
        super().__init__()
        if len(weights) > len(agents):
            raise ValueError("More weights than agents")
        if len(weights) < len(agents) and sum(weights) > 1:
            raise ValueError("Unable to infer missing weights because sum of weights > 1")
        if len(weights) < len(agents):
            missing = len(agents) - len(weights)
            missing_weight = (1 - sum(weights)) / missing
            weights = weights + [missing_weight] * missing
        norm = sum(weights)
        self.wiegihts = [ round(w / norm,2) for w in weights ]
        self.agents = agents
        self.rng = np.random.default_rng()
    def act(self, observation: any) -> int:
        agent = self.rng.choice(self.agents, p=self.wiegihts)
        return agent.act(observation)
    def __str__(self):
        return "CompoundAgent("+str({agent.__class__.__name__: weight for agent, weight in zip(self.agents, self.wiegihts)})+")"
    def __repr__(self):
        return self.__str__()

class WeightedActionAgent(Agent):
    """An agent that selects actions from a fixed set with given probabilities.

    This agent maintains a list of possible actions and their associated probability weights.
    When asked to act, it randomly selects an action according to the specified weights.

    Args:
        actions (list[Any]): List of possible actions the agent can take
        weights (list[Any], optional): Probability weights for selecting each action. Will be
            normalized to sum to 1. If fewer weights than actions are provided, remaining
            weights will be distributed evenly from the remaining probability mass.
            If None, actions will be selected with equal probability.
    """
    def __init__(self, actions: list[Any], weights: list[Any]=None):
        super().__init__()
        if weights is None:
            weights = []
        if len(weights) > len(actions):
            raise ValueError("More weights than actions")
        if len(weights) < len(actions) and sum(weights) > 1:
            raise ValueError("Unable to infer misssing weights because sum of weights > 1")
        if len(weights) < len(actions):
            missing = len(actions) - len(weights)
            missing_weights = (1-sum(weights)) / missing
            weights = weights + [missing_weights] * missing
        norm = sum(weights)
        self.weights = [w/norm for w in weights]
        self.actions = actions
        self.rng = np.random.default_rng()
    def act(self, observation):
        return self.rng.choice(self.actions,p=self.weights)
    def __str__(self):
        return f"WeightedActionAgent(actions={self.actions}, weights={self.weights})"
    def __repr__(self):
        return str(self)

class FixedActionAgent(Agent):
    """An agent that always returns the same fixed action regardless of observation.

    This agent is useful for testing or as a component in more complex agent compositions.
    It ignores the observation and always returns the action it was initialized with.

    Args:
        action (Any): The fixed action that this agent will always return
    """
    def __init__(self, action):
        super().__init__()
        self.action = action
    def act(self, observation):
        return self.action
    def __str__(self):
        return f"FixedActionAgent(action={self.action})"
    def __repr__(self):
        return str(self)

class StickyAgent(Agent):
    """A wrapper agent that makes another agent's actions "sticky" with some probability.

    This agent wraps another agent and adds persistence to its actions. Once an action
    is chosen, it will be repeated with probability equal to the sticky_factor rather
    than asking the underlying agent for a new action.

    Args:
        agent (Agent): The base agent whose actions will be made sticky
        sticky_factor (float): Probability (between 0 and 1) of repeating the last action
            instead of getting a new action from the base agent
    """
    def __init__(self, agent: Agent, sticky_factor: float):
        super().__init__()
        self.agent = agent
        self.sticky_factor = sticky_factor
        self.last_action = None
        self.rng = np.random.default_rng()
    def act(self, observation):
        if self.last_action is None or self.rng.random() > self.sticky_factor:
            self.last_action = self.agent.act(observation)
        return self.last_action
    def __str__(self):
        return f"StickyAgent(agent={self.agent}, sticky_factor={self.sticky_factor})"
    def __repr__(self):
        return str(self)

# Other potential agents
# * Markov agent (Markov chain of last action to next action)