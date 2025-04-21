from templateRL import Agent
import gymnasium as gym
import time

env = gym.make("LunarLander-v3")
class RandomAgent(Agent):
    def act(self, observation):
        return env.action_space.sample()
agent = RandomAgent()
n = 5000

factory = lambda: gym.make("LunarLander-v3")

print("Using a Random Agent")

print()
print("Sequential", n, "runs")
s = time.monotonic()
agent.play_n_episodes_sequential(factory, n, True)
print(time.monotonic()-s, 'seconds')
env.close()

# print()
# print("threaded parallel", n, "runs")
# s = time.monotonic()
# agent.play_n_episodes_parallel_threaded(factory, n, show_prog=True)
# print(time.monotonic()-s, 'seconds')

print()
print("process parallel", n, "runs")
s = time.monotonic()
agent.play_n_episodes_in_process_pool(factory, n, show_prog=True)
print(time.monotonic()-s, 'seconds')


