from templateRL import Agent
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

class RandomAgent(Agent):
    def act(self, observation):
        return env.action_space.sample()

agent = RandomAgent()
agent.play_episode(env, observation)
env.close()