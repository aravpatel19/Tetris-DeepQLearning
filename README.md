# Deep Q-Learning Tetris Agent
https://github.com/user-attachments/assets/cc993dab-b3ce-4ef2-a9cb-612e02ece5ef

## Introduction
I decided to challenge myself by creating a self-learning agent for Tetris. Leveraging Python’s extensive tools like PyTorch, I trained a model to autonomously learn and play Tetris. By the end of this project, my model could play Tetris with a level of skill comparable to a professional player.

## Objectives
- Build a functioning reinforcement learning (RL) game agent.
- Use Q-Learning to optimize gameplay.
- Train the agent to clear 1,000+ lines.

## Methodology
### 1. Environment Setup
I started by creating a `Tetris.py` file that defined the game environment. This included functions to:
- Represent tetrominoes (Tetris pieces).
- Maintain state properties.
- Render the game for visualization.

### 2. Deep Q-Network (DQN) Architecture
I connected layers of my Deep Q-Network using rectifier activation functions (ReLU). These functions mimic the behavior of biological neurons, allowing the network to process data efficiently.

### 3. Training the Agent
I employed a Deep Q-Network (DQN) for training, which is an advanced form of Q-learning. Unlike traditional algorithms, the DQN allowed the agent to learn by trial and error, using rewards as motivation to improve its strategy.

### 4. Iterative Training
I trained the model for several hours, letting it learn and improve incrementally. Through repeated iterations, the agent eventually became capable of clearing over 1,000 lines consistently.

### Equation Utilized
The training process was guided by the Bellman Equation for optimal Q-values:

<img width="607" alt="Screenshot 2025-01-14 at 2 15 49 PM" src="https://github.com/user-attachments/assets/a0e487ee-f89b-4e4e-b620-d922a0f78486" />

Where:
- `s_t`: State at time `t`.
- `a_t`: Action taken at time `t`.
- `π`: Policy function determining the action given the current state.
- `r_t`: Reward obtained by taking action `a_t` in state `s_t`.
- `γ`: Discount factor.

## Results
<img width="563" alt="Screenshot 2025-01-14 at 2 08 38 PM" src="https://github.com/user-attachments/assets/107dffb8-9a39-49b5-a5da-f1da07465692" />

### Model Performance

The agent’s performance improved exponentially over time:
- During the first 1,000 epochs, the agent’s score remained stagnant as it struggled to learn.
- Once the agent began receiving rewards for clearing lines, it rapidly improved, achieving exponential growth in its performance.

The attached graph demonstrates the improvement in the agent’s average score over epochs.

## Research and Analysis
Furthermore, I conducted extensive research on potential methods and implementations. A key source of inspiration was the academic paper "Playing Tetris with Deep Reinforcement Learning" by Matt Stevens and Sabeek Pradhan from Stanford.

### Future Improvements
If I were to revisit this project, I would:
1. Allocate more time to train the game agent for even better performance.
2. Develop a more accessible interface for running and visualizing the game agent, such as a web-based application.

## Conclusion
Through this project, I gained valuable experience in reinforcement learning and neural network design. Building a Deep Q-Learning Tetris agent was both challenging and rewarding, showcasing the potential of AI to master complex tasks autonomously.

