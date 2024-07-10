import os 

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
print('base dir is',BASE_DIR)
TRAIN_CHECKPOINT = os.path.join(BASE_DIR,'best_model.pth')#check this
print(TRAIN_CHECKPOINT)
RGB_DIR = os.path.join(BASE_DIR,'rgb_images')
SEGMENTATION_DIR = os.path.join(BASE_DIR,'segmentation_images')
INPUT_VIDEO = os.path.join(BASE_DIR,'test_video1.mp4')#check this
OUTPUT_DIR = os.path.join(BASE_DIR,'Ransac_output')
INPUT_DIR = os.path.join(BASE_DIR,'Ransac_output')#FOLDER FROM WHERE YOU WANT IMAGE SEQUENCE
OVERLAY_DIR = os.path.join(BASE_DIR,'Segmentation_overlay')
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR,'output_video.mp4')#check this
