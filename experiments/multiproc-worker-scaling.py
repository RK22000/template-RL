from templateRL import RandomAgent
import gymnasium as gym
import time
import mlflow

factory = lambda: gym.make("LunarLander-v3")
action_size=4

ra = RandomAgent(action_size)
n = 10000
m = 50

EXPERIMENT_NAME = "Multiproc-Worker-Scaling"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"Starting experiment {EXPERIMENT_NAME} up to {m} workers")
for n_workers in range(1, m+1):
    with mlflow.start_run():
        mlflow.log_params({
            "number of rollouts": n,
            "number of workers": n_workers,
        })
        s = time.monotonic()
        ra.play_n_episodes_in_process_pool(factory, n, n_workers, show_prog=False)
        t = time.monotonic()
        t -= s
        print(f"{n: >5} rollouts {n_workers: >5} workers {round(t): >5} seconds")
        mlflow.log_metric("runtime", t)







