import os
import shutil
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from collections import deque
from random import random, randint, sample

# Function to initialize the TensorBoard writer
def initialize_logging(log_path: str) -> SummaryWriter:
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    return SummaryWriter(log_path)

# Function to initialize the optimizer and criterion
def initialize_optimizer_and_criterion(model: nn.Module, learning_rate: float):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    return optimizer, criterion

# Function to initialize replay memory
def initialize_replay_memory(size: int) -> deque:
    return deque(maxlen=size)

# Function to handle environment state resets
def reset_environment_state(env, model) -> torch.Tensor:
    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()
    return state

# Function to compute epsilon value for exploration-exploitation
def calculate_epsilon(initial_epsilon: float, final_epsilon: float, decay_epochs: int, epoch: int) -> float:
    return final_epsilon + max(decay_epochs - epoch, 0) * (initial_epsilon - final_epsilon) / decay_epochs

# Function to choose the next action using epsilon-greedy strategy
def choose_action(model: nn.Module, states: torch.Tensor, epsilon: float) -> int:
    u = random()
    if u <= epsilon:
        return randint(0, states.shape[0] - 1)  # Random action
    else:
        model.eval()
        with torch.no_grad():
            predictions = model(states)[:, 0]
        model.train()
        return torch.argmax(predictions).item()  # Exploit with best prediction

# Function to handle batch sampling from replay memory
def sample_batches(replay_memory, batch_size: int):
    batch = sample(replay_memory, min(len(replay_memory), batch_size))
    state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    state_batch = torch.stack(tuple(state for state in state_batch))
    reward_batch = torch.from_numpy(torch.tensor(reward_batch, dtype=torch.float32)[:, None].numpy())
    next_state_batch = torch.stack(tuple(state for state in next_state_batch))
    return state_batch, reward_batch, next_state_batch, done_batch
