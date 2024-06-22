import math

import gym
import numpy as np
import random
from collections import deque

env = gym.make('CartPole-v1', render_mode="human")


# identifying the parameters

learning_rate = 0.1
discount_rate = 0.9
exploration_rate = 0.05
max_steps = 1000
episodes = 3000

q_table = np.zeros((40, 2))


# Q-learning algorithm
part_ang = (env.observation_space.high[2] - env.observation_space.low[2]) / 40


# desctitization of the state which is the angle :
def discritization_ang (state , part_ang):
    n = (state - env.observation_space.low[2]) / part_ang
    return math.ceil(n) - 1

# start of the training phase :
for episode in range(episodes):
    print("Episodes : ", episode)
    state = env.reset()
    state1 = discritization_ang(state[0][2], part_ang)
    state = state1
    done = False
    totReward = 0
    for step in range(max_steps):
        if random.uniform(0, 1) < exploration_rate:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        next_state, reward, k, truncated, info = env.step(action)

        totReward = totReward + reward

        if ((next_state[2] > 0.19) or (next_state[2] < -0.19)):
            done = True
        next_state1 = discritization_ang(next_state[2], part_ang)
        next_state = next_state1

        # Update Q-value
        q_value = q_table[state][action]
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + discount_rate * q_table[next_state][best_next_action]
        q_table[state][action] = q_value + learning_rate * (td_target - q_value)

        state = next_state

        if done:
            break


    print("reward =  ", totReward)

print("Training finished.\n")

print(q_table)

env.close()