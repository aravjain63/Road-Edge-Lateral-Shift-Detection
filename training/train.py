#imports
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
from collections import defaultdict
from model import *
from utils import *
from dataset import *
from transforms import *
from paths import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time



train_dataset = TuneDataset(
    image_dir=tune_images_path_train,
    transform=train_transform,
    test=False
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
    collate_fn=custom_collate

)

val_dataset = TuneDataset(
    image_dir=tune_images_path_valid,
    transform=val_transforms,
    test=False
)
val_loader = DataLoader(
    val_dataset,  # Changed from train_dataset to val_dataset
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
    collate_fn=custom_collate
  # Usually, we don't shuffle validation data
)

test_dataset = TuneDataset(
    image_dir=tune_images_path_test,
    transform=val_transforms,  # Changed from None to val_transforms
    test=True
)
test_loader = DataLoader(  # Changed from val_loader to test_loader
    test_dataset,  # Changed from train_dataset to test_dataset
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
    collate_fn=custom_collate
  # We don't shuffle test data
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# Sanity check: Look at a batch from each loader
for i, (images, masks) in enumerate(train_loader):
    print(f"Train batch {i}: image shape {images.shape}, mask shape {masks.shape}")
    break

for i, (images, masks) in enumerate(val_loader):
    print(f"Validation batch {i}: image shape {images.shape}, mask shape {masks.shape}")
    break

for i, (images, masks) in enumerate(test_loader):
    print(f"Test batch {i}: image shape {images.shape}, mask shape {masks.shape}")
    break



EPOCHS=EPOCHS
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = IoULoss().to(DEVICE)
# loss_fn =nn.BCELOSS().to(DEVICE)
LEARNING_RATE=1e-3

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

torch.cuda.empty_cache()
model.load_state_dict(torch.load(os.path.join(BASE_DIR,'best_model.pth')))

for param in model.downs.parameters():
    param.requires_grad = False


history, log_dir, checkpoint_dir = train_model(model, train_loader, val_loader, loss_fn, optimizer, DEVICE, epochs=EPOCHS, patience=10)
print(f"Training complete. Log directory: {log_dir}")
print(f"Checkpoints directory: {checkpoint_dir}")

model.load_state_dict(torch.load(os.path.join(BASE_DIR,checkpoint_dir,'best_model.pth')))

metrics = evaluate_model(model, test_dataset, DEVICE)
print("Model Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Visualize some predictions
visualize_predictions(model, test_dataset, DEVICE)

