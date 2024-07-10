from curve_fitting.config import *
from curve_fitting.curve_fit import *
from curve_fitting.extract_frames import *
from curve_fitting.saving_images import *
from training.prediction import *
from config import *
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from matplotlib import pyplot as plt
from collections import defaultdict
import random
import os
from tqdm import tqdm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from training.model import*
import os
from training.paths import *
from training.transforms import test_transforms
from utils.image_to_video import *

print('here')
rgb_dir=RGB_DIR
# all directories are created automatically if they dont exist
os.makedirs(RGB_DIR,exist_ok=True)
os.makedirs(SEGMENTATION_DIR,exist_ok=True)
os.makedirs(OVERLAY_DIR,exist_ok=True)
os.makedirs(OUTPUT_DIR,exist_ok=True)

# extract frames from input_video to rgb_dir
FrameCapture(INPUT_VIDEO,RGB_DIR,sample_rate=5) #check this

# create segmentation images for image frames
predict_image(RGB_DIR,out_path=SEGMENTATION_DIR)
# run ransac algo
# set create_excel and create_images accordingly
findDistance(rgb_dir,SEGMENTATION_DIR,output_dir=OUTPUT_DIR,create_excel=False,create_images=True)
# visualise segmentation mask
overlay(rgb_dir,SEGMENTATION_DIR,output_dir=OVERLAY_DIR)

# save data in excel
Excel(rgb_dir,SEGMENTATION_DIR,output_dir=OUTPUT_DIR)

# convert ransac photos to images
create_video_from_images(OUTPUT_DIR, OUTPUT_VIDEO_PATH, fps=30)


