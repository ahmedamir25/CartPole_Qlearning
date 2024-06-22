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


# put here the Q_table that  you got from the training code
q_table = []


# Q-learning algorithm :
part_ang = (env.observation_space.high[2] - env.observation_space.low[2]) / 40

# desctitization of the state which is the angle :
def discritization_ang (state , part_ang):
    n = (state - env.observation_space.low[2]) / part_ang
    return math.ceil(n) - 1


# start the testing phase :
for episode in range(50):
    state = env.reset()
    state1 = discritization_ang(state[0][2], part_ang)
    state = state1
    done = False
    total_reward = 0

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state])
        next_state, reward, k, truncated, info = env.step(action)
        if ((next_state[2] > 0.19) or (next_state[2] < -0.19)):
            done = True
        else:
            done = False
        next_state1 = discritization_ang(next_state[2], part_ang)
        next_state = next_state1
        state = next_state
        total_reward += reward
        if done:
            break

    print(f"Episode: {episode}, Total Reward: {total_reward}")


env.close()