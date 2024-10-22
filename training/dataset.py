import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import logging
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

logging.basicConfig(level=logging.INFO)


class TuneDataset(Dataset):
    def __init__(self, image_dir, transform=None, test=False):
        self.image_dir = image_dir
        self.test = test
        self.transform = transform
        self.images = []
        for f in os.listdir(image_dir):
            if not f.endswith('_mask.png') and not f.startswith('.'):#check this
                if os.path.isfile(os.path.join(image_dir, f)):
                    self.images.append(f)
        logging.info(f"Found {len(self.images)} images in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")

        
        mask_name = img_name.replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.image_dir, mask_name)
        try:
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        except Exception as e:
            logging.error(f"Error loading mask {mask_path}: {str(e)}")
            raise RuntimeError(f"Error loading mask {mask_path}: {str(e)}")

        if self.transform is not None:
            try:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            except Exception as e:
                logging.error(f"Error applying transform to {img_path}: {str(e)}")
                raise RuntimeError(f"Error applying transform to {img_path}: {str(e)}")

        return image, mask

    def verify_dataset(self):
        for img_name in self.images:
            img_path = os.path.join(self.image_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()
                if not self.test:
                    mask_path = os.path.join(self.image_dir, img_name.replace('.jpg', '_mask.png'))
                    with Image.open(mask_path) as mask:
                        mask.verify()
            except Exception as e:
                print(f"Corrupted file detected: {img_path}")
def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

