from test_helpers.test_helpers import (
    set_random_seed,
    load_trained_model,
    initialize_environment,
    move_model_to_gpu,
    select_best_action
)
from src.tetris import Tetris
import torch
from test_helpers.testing_config import TestConfig

def test(config: TestConfig):
    """Test the trained Tetris model using the provided configuration."""
    # Set random seeds for reproducibility
    set_random_seed()

    # Load the trained model
    model = load_trained_model(config.saved_path)
    model.eval()

    # Initialize the Tetris environment with the specified configuration
    env = initialize_environment(config.width, config.height, config.block_size)

    # Move the model to GPU if available
    move_model_to_gpu(model)

    # Loop to play the game automatically using the trained model
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)

        # Get predictions and select the best action
        index = select_best_action(model, next_states, next_actions)
        action = next_actions[index]

        # Take action in the environment and record the game
        _, done = env.step(action, render=True)

        # Stop recording if the game is over
        if done:
            break
        
if __name__ == "__main__":
    # Create an instance of TestConfig and pass it to the test function
    config = TestConfig()
    test(config)
