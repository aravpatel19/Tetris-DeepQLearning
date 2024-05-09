class TetrisTrainingConfig:
    def __init__(self):
        """training configuration with hardcoded default values."""
        # tetris environment dimensions
        self.width = 10
        self.height = 20
        self.block_size = 30
        # training parameters
        self.batch_size = 512
        self.lr = 1e-3  # learning rate
        self.gamma = 0.99  # discount factor for future rewards
        self.initial_epsilon = 1  # exploration rate start value
        self.final_epsilon = 1e-3  # exploration rate end value
        self.num_decay_epochs = 2000  # epochs for epsilon decay
        self.num_epochs = 3000  # total training epochs
        self.save_interval = 1000  # epoch interval to save the model
        self.replay_memory_size = 30000  # maximum replay memory size
        # logging and model saving paths
        self.log_path = "tensorboard"
        self.saved_path = "trained_models"
