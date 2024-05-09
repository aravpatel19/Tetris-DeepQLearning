class TestConfig:
    def __init__(self):
        """initialize the testing configuration with hardcoded default values."""
        # tetris environment settings
        self.width = 10
        self.height = 20
        self.block_size = 30
        # testing and output parameters
        self.fps = 300  # frames per second for video output
        self.saved_path = "trained_models"  # directory containing trained models
        self.output = "output.mp4"  # output video filename
