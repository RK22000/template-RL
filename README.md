# Template RL

This is a project for easily creating and using custom RL agents.

## Quick Set up

```sh
git clone https://github.com/RK22000/template-RL.git
cd template-RL
# Pick your python as desired
# I'm using 3.11
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -Ur requirements.txt
```

Once set up have a look at or try running the [ppo3 script](scripts/ppo3.py) to get a feel for how to build on the template-RL framework.

```sh
python scripts/ppo3.py
```

This project uses mlflow for logging so start an mlflow ui server to view the logs during and after a script or an experiment.

```sh
mlflow ui
```

## Motivation

Reinforcement Learning defines a nice abstration of an agent taking observations from the environment and taking actions to receive some reward.

[![image sourced from https://www.educba.com/what-is-reinforcement-learning/](https://cdn.educba.com/academy/wp-content/uploads/2019/11/reinforcement3.png)](https://www.educba.com/what-is-reinforcement-learning/)

I wanted to make a coding framework that allows me to implement algorithms while sticking as close as possible to the RL abstraction

## Notes of demos and scripts

```sh
python scripts/ppo-agent-demo.py
```

[save-load-demo.py](scripts/save-load-demo.py) is a script that will

1. Try to load a specified ppo agent file as a `PPOAgent`
2. If the load fails it will train and save a fresh `PPOAgent`. Then this save file will be loaded as a `PPOAgent`
3. The loaded `PPOAgent` will be used to collect rollouts.
4. Everything is logged using mlflow. So in a separate terminal run `mlflow ui` to see the mlflow logs during or after the script run.

```sh
python scripts/ppo-agent-demo.py
```

[ppo-agent-demo.py](scripts/ppo-agent-demo.py) is a script that will showcase a specified trained agent file on the Lunar landing environment.
