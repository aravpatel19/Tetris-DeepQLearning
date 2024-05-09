class TetrisTrainingConfig:
    def __init__(self):
        """Training configuration with hardcoded default values."""
        # Tetris environment dimensions
        self.width = 10
        self.height = 20
        self.block_size = 30
        # Training parameters
        self.batch_size = 512
        self.lr = 1e-3  # Learning rate
        self.gamma = 0.99  # Discount factor for future rewards
        self.initial_epsilon = 1  # Exploration rate start value
        self.final_epsilon = 1e-3  # Exploration rate end value
        self.num_decay_epochs = 2000  # Epochs for epsilon decay
        self.num_epochs = 3000  # Total training epochs
        self.save_interval = 1000  # Epoch interval to save the model
        self.replay_memory_size = 30000  # Maximum replay memory size
        # Logging and model saving paths
        self.log_path = "tensorboard"
        self.saved_path = "trained_models"
        
