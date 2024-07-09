import os 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
RGB_DIR = os.path.join(BASE_DIR,'rgb_images')
SEGMENTATION_DIR = os.path.join(BASE_DIR,'segmentation_images')
INPUT_VIDEO = os.path.join(BASE_DIR,'video.avi')
OUTPUT_DIR = os.path.join(BASE_DIR,'Ransac_output')
OVERLAY_DIR = os.path.join(BASE_DIR,'Segmentation_overlay')