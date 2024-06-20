import math

import gym
import numpy as np
import random
from collections import deque

env = gym.make('CartPole-v1', render_mode="human")



learning_rate = 0.1
discount_rate = 0.9
exploration_rate = 0.05
max_steps = 500
episodes = 500

q_table = np.zeros((100, 2))

# Q-learning algorithm

part_pos = (env.observation_space.high[0] - env.observation_space.low[0]) / 10
part_ang = (env.observation_space.high[2] - env.observation_space.low[2]) / 10


def discritization_pos (state , part_pos):
    n = (state - env.observation_space.low[0]) / part_pos
    return math.ceil(n) - 1

def discritization_ang (state , part_ang):
    n = (state - env.observation_space.low[2]) / part_ang
    return math.ceil(n) - 1


for episode in range(episodes):
    state = env.reset()
    state0 = discritization_pos(state[0][0], part_pos)
    state1 = discritization_ang(state[0][2], part_ang)
    state = state0*10 + state1
    done = False
    for step in range(max_steps):
        if random.uniform(0, 1) < exploration_rate:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        next_state, reward, done, truncated, info = env.step(action)
        next_state0 = discritization_pos(next_state[0], part_pos)
        next_state1 = discritization_ang(next_state[2], part_ang)
        next_state = next_state0*10 + next_state1

        # Update Q-value
        q_value = q_table[state][action]
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + discount_rate * q_table[next_state][best_next_action]
        q_table[state][action] = q_value + learning_rate * (td_target - q_value)

        state = next_state

        if done:
            break

    # exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

print("Training finished.\n")


for episode in range(50):
    state = env.reset()
    state0 = discritization_pos(state[0][0], part_pos)
    state1 = discritization_ang(state[0][2], part_ang)
    state = state0 * 10 + state1
    done = False
    total_reward = 0

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state])
        next_state, reward, done, truncated, info = env.step(action)
        next_state0 = discritization_pos(next_state[0], part_pos)
        next_state1 = discritization_ang(next_state[2], part_ang)
        next_state = next_state0 * 10 + next_state1
        state = next_state
        total_reward += reward
        if done:
            break

    print(f"Episode: {episode}, Total Reward: {total_reward}")

env.close()
