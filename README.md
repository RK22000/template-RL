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

Once set up try running the [ppo3 script](scripts/ppo3.py).

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
