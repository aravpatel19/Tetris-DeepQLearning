class TestConfig:
    def __init__(self):
        """Initialize the testing configuration with hardcoded default values."""
        # Tetris environment settings
        self.width = 10
        self.height = 20
        self.block_size = 30
        # Testing and output parameters
        self.fps = 300  # Frames per second for video output
        self.saved_path = "trained_models"  # Directory containing trained models
        self.output = "output.mp4"  # Output video filename
