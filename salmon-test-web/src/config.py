"""
Configuration settings for fish scale classification
"""

# S3 Settings
BUCKET = "uw-fish-scale-data-879554841091-us-west-2"
FILE_PREFIX = "Kalama Spring Chinook AI Project"
CSV_KEY = "Kalama_GIX_Data_Final.csv"

# Label Mapping
LABEL_MAP = {"Hatchery": 0, "Natural": 1}

# Data Split Ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Image Processing Parameters
IMG_SIZE = 224  # Higher resolution for better texture detail preservation
# Note: Original images are 1920x1080 (compression: ~3.5x)
# For ablation study, try: [224, 320, 384, 448, 512]
# 512 preserves more fine-grained scale patterns but requires more GPU memory
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# Image Preprocessing Pipeline Parameters
ROBUST_NORM_PERCENTILES = (1, 99)  # p1, p2
DESPECKLE_MEDIAN_KERNEL = 5
DESPECKLE_MORPH_RADIUS = 2
FFT_RLO_FRAC = 0.015
FFT_RHI_FRAC = 0.22
FFT_SOFTNESS = 6.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
LAPLACIAN_WEIGHT = 0.15

# Training Parameters
BATCH_SIZE = 64
NUM_WORKERS = 8  
NUM_EPOCHS = 50  # Reduced from 50 since early stopping usually triggers around epoch 20-27
LEARNING_RATE = 1e-5  # Reduced to prevent overfitting
PATIENCE = 3  # Early stopping patience

# Model Parameters
MODEL_NAME = "resnet18"  # ResNet18 is better for this data size
NUM_CLASSES = 2
FREEZE_BACKBONE = False  # Set to True to only train classifier layer

# Training Augmentation
TRAIN_HORIZONTAL_FLIP = False
TRAIN_ROTATION_DEGREES = 10
TRAIN_COLOR_JITTER_CONTRAST = 0.2
TRAIN_COLOR_JITTER_BRIGHTNESS = 0.2

# Output Paths
RESULTS_DIR = "../results"
CHECKPOINT_PATH = "../results/checkpoints/best_18_50_aug_v2.ckpt"
MODEL_SAVE_PATH = "../results/models/resnet18_scale_classifier_50_aug_v2.pth"
TRAINING_HISTORY_PATH = "../results/training_logs/training_history_18_50_aug_v2.png"
TRAINING_RESULTS_DIR = "../results/training_logs_18_50_aug_v2"
VISUALIZATION_DIR = "../results/visualizations_18_50_aug_v2"
EVALUATION_DIR = "../results/evaluations_18_50_aug_v2"
