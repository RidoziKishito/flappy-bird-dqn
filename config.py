# ====== CONFIG.PY ======

# Screen dimensions and game settings
SCREEN_WIDTH = 400        # Width of game window in pixels
SCREEN_HEIGHT = 600       # Height of game window in pixels 
FPS = 60                 # Frames per second

# Training parameters
EPI_NUMS = 2000          # Number of episodes for training
ASSET_DIR_NAME = "assets"  # Directory containing game assets
CHECKPOINT_PATH = "best.pth"  # Path to save/load model weights

# Game layout constants
INIT_PIPE_OFFSET = 100   # Initial distance of first pipe from screen edge
MIN_GAP_Y = 50          # Minimum y-position for pipe gap
GROUND_HEIGHT = 100     # Height of ground from bottom of screen

# Physics constants
GRAVITY = 0.5           # Acceleration due to gravity
FLAP_VEL = -9.0        # Upward velocity when flapping
MAX_VEL = 12.0         # Maximum vertical velocity

# Reward system parameters
LIVING_REWARD = 0.01           # Small reward for staying alive
SCORE_REWARD = 10.0           # Reward for passing through pipes
DEATH_PENALTY = -10.0         # Penalty for dying
VERTICAL_WEIGHT = 0.3         # Weight for vertical position alignment
VELOCITY_WEIGHT = 0.1         # Weight for velocity control
CENTER_BONUS_MULT = 5.0       # Multiplier for centering in pipe gap
APPROACHING_THRESHOLD = 0.3    # Distance threshold for approaching pipe
APPROACHING_MULTIPLIER = 2.0   # Reward multiplier when near pipes

# Animation settings
ANIM_FREQ = 5           # Animation frame update frequency

# Bird parameters
BIRD_X_POS = 80        # Fixed x position of bird
BIRD_RADIUS = 12       # Collision radius of bird
PIPE_WIDTH = 52        # Width of pipe obstacles