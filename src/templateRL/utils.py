def rewards_to_cumilative_rewards(rewards: list[float], gamma: float):
    cum_rewards = [rewards.pop()]
    while rewards:
        cum_rewards.append(gamma*cum_rewards[-1] + rewards.pop())  
    cum_rewards = reversed(cum_rewards)
    return cum_rewards