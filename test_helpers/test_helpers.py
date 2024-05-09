import torch
import cv2
from src.tetris import Tetris

def set_random_seed():
    """set a fixed random seed for reproducibility."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

def load_trained_model(saved_path: str):
    """load the trained tetris model from the specified path."""
    if torch.cuda.is_available():
        return torch.load(f"{saved_path}/tetris_1883")
    else:
        return torch.load(f"{saved_path}/tetris_1883", map_location=lambda storage, loc: storage)

def initialize_environment(width: int, height: int, block_size: int) -> Tetris:
    """initialize and reset the tetris game environment."""
    env = Tetris(width=width, height=height, block_size=block_size)
    env.reset()
    return env

def move_model_to_gpu(model: torch.nn.Module):
    """move the model to gpu if available."""
    if torch.cuda.is_available():
        model.cuda()

def select_best_action(model: torch.nn.Module, states: torch.Tensor, actions: list) -> int:
    """select the best action based on model predictions."""
    if torch.cuda.is_available():
        states = states.cuda()
    predictions = model(states)[:, 0]
    return torch.argmax(predictions).item()
