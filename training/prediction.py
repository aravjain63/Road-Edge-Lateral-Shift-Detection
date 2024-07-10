import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from training.model import *
import logging
import os
from training.paths import *
from training.transforms import test_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET().to(DEVICE)
model.load_state_dict(torch.load(TRAIN_CHECKPOINT))

def predict_image(path,out_path,transform=test_transforms,model=model):
    images = []
    for f in os.listdir(path):
        if not f.endswith('_mask.png') and not f.startswith('.'):
            if os.path.isfile(os.path.join(path, f)):
                images.append(f)
    for index in range(len(images)):
        img_name = images[index]
        img_path = os.path.join(path, img_name)

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            height = image.shape[0]
            # FOR NON ROBOFLOW DATA
            cropped_image = image[:int(height * 0.80), :]
            augmented = transform(image=cropped_image)
            image = augmented['image'].unsqueeze(0).to(DEVICE)
            print(image.shape)
            with torch.no_grad():
                model.eval()
                output=model(image)
                preds1 = (output>0.8).float()
                # print(preds.shape)
                # torch.Size([1, 1, 128, 128])
            binary_map1 = (preds1[0].squeeze().cpu().numpy()).astype(np.uint8) * 255
            # //not resized again,further calculations are done on 75% if the image only
            # road_height = int(binary_map1.shape[0] * 0.25)
            # binary_map1[-road_height:, :] = 0
            cv2.imwrite(os.path.join(out_path,f'output{index}_segmentation.png'), binary_map1)
            # fig, axs = plt.subplots(4, 4, figsize=(15, 5))
            # input_np = image[0].cpu().numpy().transpose(1, 2, 0)
            # pred_np = preds[0].cpu().numpy().squeeze()
            # axs[0, 0].imshow(input_np)
            # axs[0, 0].set_title('Input Image')
            # axs[0, 2].imshow(pred_np, cmap='gray')
            # axs[0, 2].set_title('Prediction')
            # plt.tight_layout()
            # plt.show()
        except Exception as e:
            logging.error(f"Error loading image {path}: {str(e)}")
            raise RuntimeError(f"Error loading image {path}: {str(e)}")
    
# path = PREDICTION_PATH
# predict_image_image(path,test_transforms,model)








