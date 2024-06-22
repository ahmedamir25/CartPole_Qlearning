# CartPole Q-Learning:

## CartPole Q-Learning Implementation :
This project implements a Q-learning algorithm to solve the CartPole-v1 environment using OpenAI's Gym.

## Overview :
The core idea is to use Q-learning, a reinforcement learning algorithm, to train an agent to balance a pole on a cart. The unique aspect of this implementation is the discretization of the pole's angle into 40 parts, which simplifies the state space for training.

## Key Concepts : 
### Q-learning:
A model-free reinforcement learning algorithm used to find the optimal action-selection policy for a given finite Markov decision process.
### State Discretization:
The continuous state space (specifically the pole's angle) is divided into 40 discrete parts. This simplifies the problem and makes it more tractable for Q-learning.
### Angle as State:
Only the pole's angle is used as the state representation, ignoring other variables for simplicity.

## Training Phase:
During training, the agent explores and exploits the environment based on the Q-learning algorithm:

1. It starts with a random state.
2. For each step, it either takes a random action (explore) or the best-known action (exploit).
3. The Q-table is updated based on the reward received and the future reward estimates.
The training process continues for 3000 episodes, where each episode can last up to 1000 steps unless the pole falls.

## Testing Phase:
After training, the code includes a section to test the performance of the trained agent. This phase allows observing how well the agent has learned to balance the pole.

## Code Execution:
To run the training and testing phases, simply execute the provided script. Ensure that the necessary dependencies, such as Gym and NumPy, are installed.

## Results:
Upon completion, the Q-table containing the learned values will be printed, reflecting the agent's knowledge of the optimal actions for different states.
so copy and paste that Q-table and put it in the testing code to see the results

## Further Information
For more information about Q-learning and the CartPole environment, visit the following links:
- https://www.gymlibrary.dev/content/basic_usage/
- https://www.gymlibrary.dev/environments/classic_control/cart_pole/
- https://blog.paperspace.com/getting-started-with-openai-gym/
