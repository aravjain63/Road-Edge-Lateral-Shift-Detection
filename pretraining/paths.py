import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
KITTI_DIR = os.path.join(BASE_DIR, 'kitti')
TRAINING_DIR = os.path.join(KITTI_DIR, 'training')
MASK_PATH = os.path.join(TRAINING_DIR, 'gt_image_2')
IMG_PATH = os.path.join(TRAINING_DIR, 'image_2')
PRETRAIN_PATH = os.path.join(BASE_DIR, 'pretraining')
# Training parameters
EPOCHS = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 4

# Print paths for debugging
print(f"BASE_DIR: {BASE_DIR}")
print(f"KITTI_DIR: {KITTI_DIR}")
print(f"TRAINING_DIR: {TRAINING_DIR}")
print(f"MASK_PATH: {MASK_PATH}")
print(f"IMG_PATH: {IMG_PATH}")

# Check if directories exist
print(f"KITTI_DIR exists: {os.path.exists(KITTI_DIR)}")
print(f"TRAINING_DIR exists: {os.path.exists(TRAINING_DIR)}")
print(f"MASK_PATH exists: {os.path.exists(MASK_PATH)}")
print(f"IMG_PATH exists: {os.path.exists(IMG_PATH)}")