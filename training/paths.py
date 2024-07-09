import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
tune_images_path_train = os.path.join(BASE_DIR,'roboflow','train')
tune_images_path_valid = os.path.join(BASE_DIR,'roboflow','valid')
tune_images_path_test = os.path.join(BASE_DIR,'roboflow','test')
TRAIN_CHECKPOINT = os.path.join(BASE_DIR,'training','best_model.pth')
print('train_checkpoint',TRAIN_CHECKPOINT)
# change these
# PREDICTION_PATH = os.path.join(BASE_DIR,'roboflow','test')
# PREDICTION_OUTPUT_PATH = os.path.join(BASE_DIR,'roboflow','outputs')

# Training parameters
EPOCHS = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 4

# Print paths for debugging
print(f"BASE_DIR: {BASE_DIR}")
print(f"TRAINING_DIR: {tune_images_path_train}")
print(f"validation: {tune_images_path_valid}")
print(f"test: {tune_images_path_test}")

# Check if directories exist
print(f"TRAINING_DIR exists: {os.path.exists(tune_images_path_train)}")
print(f"validation exists: {os.path.exists(tune_images_path_valid)}")
print(f"testing exists: {os.path.exists(tune_images_path_test)}")