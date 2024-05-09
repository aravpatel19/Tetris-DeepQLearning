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
    """test the trained tetris model using the provided configuration."""
    # set random seeds for reproducibility
    set_random_seed()

    # load the trained model
    model = load_trained_model(config.saved_path)
    model.eval()

    # initialize the tetris environment with the specified configuration
    env = initialize_environment(config.width, config.height, config.block_size)

    # move the model to gpu if available
    move_model_to_gpu(model)

    # loop to play the game automatically using the trained model
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)

        # get predictions and select the best action
        index = select_best_action(model, next_states, next_actions)
        action = next_actions[index]

        # take action in the environment and record the game
        _, done = env.step(action, render=True)

        # stop recording if the game is over
        if done:
            break

if __name__ == "__main__":
    # create an instance of testconfig and pass it to the test function
    config = TestConfig()
    test(config)
