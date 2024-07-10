import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CHECKPOINT

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

# Dataset paths



tune_images_path_train = os.path.join(BASE_DIR,'roboflow','train')  #check this
tune_images_path_valid = os.path.join(BASE_DIR,'roboflow','valid') #check this
tune_images_path_test = os.path.join(BASE_DIR,'roboflow','test') #check this
TRAIN_CHECKPOINT = TRAIN_CHECKPOINT
print('train_checkpoint',TRAIN_CHECKPOINT)
# PREDICTION_PATH = os.path.join(BASE_DIR,'roboflow','test')
# PREDICTION_OUTPUT_PATH = os.path.join(BASE_DIR,'roboflow','outputs')

# Training parameters
EPOCHS = 200 #check this
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

# Print paths for debugging
print(f"BASE_DIR: {BASE_DIR}")
print(f"TRAINING_DIR: {tune_images_path_train}")
print(f"validation: {tune_images_path_valid}")
print(f"test: {tune_images_path_test}")

# Check if directories exist
print(f"TRAINING_DIR exists: {os.path.exists(tune_images_path_train)}")
print(f"validation exists: {os.path.exists(tune_images_path_valid)}")
print(f"testing exists: {os.path.exists(tune_images_path_test)}")