import torch

from train_helpers.training_config import TetrisTrainingConfig

from tensorboardX import SummaryWriter
from train_helpers.train_helpers import (
    initialize_logging,
    initialize_optimizer_and_criterion,
    initialize_replay_memory,
    reset_environment_state,
    calculate_epsilon,
    choose_action,
    sample_batches
)
from src.tetris import Tetris
from src.dqnetwork import DQNetwork

def train(config: TetrisTrainingConfig):
    """Train the Tetris Deep Q Network using the provided training configuration."""
    # Set random seeds for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # Initialize TensorBoard logging and replay memory
    writer = initialize_logging(config.log_path)
    replay_memory = initialize_replay_memory(config.replay_memory_size)

    # Initialize the Tetris environment, model, optimizer, and loss function
    env = Tetris(width=config.width, height=config.height, block_size=config.block_size)
    model = DQNetwork()
    optimizer, criterion = initialize_optimizer_and_criterion(model, config.lr)

    # Reset the game environment and obtain the initial state
    state = reset_environment_state(env, model)

    epoch = 0
    while epoch < config.num_epochs:
        # Obtain the next possible states from the current environment
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)

        if torch.cuda.is_available():
            next_states = next_states.cuda()

        # Calculate epsilon using the decay mechanism and select the next action
        epsilon = calculate_epsilon(config.initial_epsilon, config.final_epsilon, config.num_decay_epochs, epoch)
        action_index = choose_action(model, next_states, epsilon)
        next_state = next_states[action_index, :]
        action = next_actions[action_index]

        # Execute the action in the environment and receive the reward and terminal status
        reward, done = env.step(action, render=True)

        if torch.cuda.is_available():
            next_state = next_state.cuda()

        # Store the experience in the replay memory
        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = reset_environment_state(env, model)
        else:
            state = next_state
            continue

        # Skip training if the replay memory is not sufficiently filled
        if len(replay_memory) < config.replay_memory_size / 10:
            continue

        # Increment epoch count and prepare the training batch
        epoch += 1
        state_batch, reward_batch, next_state_batch, done_batch = sample_batches(replay_memory, config.batch_size)

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # Obtain Q-values for the current state batch
        q_values = model(state_batch)

        # Predict Q-values for the next state batch (target values)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        # Compute the target values (y_batch)
        y_batch = torch.cat(
            tuple(reward if done else reward + config.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        # Backpropagation: update the model's weights based on the loss
        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        # Log training progress
        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch, config.num_epochs, action, final_score, final_tetrominoes, final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        # Save the model periodically or if the final cleared lines exceed a threshold
        if epoch > 0 and epoch % config.save_interval == 0 or final_cleared_lines > 100:
            torch.save(model, "{}/tetris_{}".format(config.saved_path, epoch, final_cleared_lines))

    # Final model save
    torch.save(model, "{}/tetris".format(config.saved_path))



if __name__ == "__main__":
    # Instantiate the training configuration
    config = TetrisTrainingConfig()
    # Start training the model with the provided configuration
    train(config)