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
history = defaultdict(list)
from model import *
from utils import *
from dataset import *
from augmentation import augmentation
from paths import *


mask_path=MASK_PATH
img_path=IMG_PATH





train_set_size = int(len(os.listdir(img_path)) * 0.8)
print(f"Number of Training Examples: {train_set_size}")

validation_set_size = int(len(os.listdir(img_path)) * 0.1)
print(f"Number of Validation Examples: {validation_set_size}")

test_set_size = int(len(os.listdir(img_path)) - train_set_size - validation_set_size)
print(f"Number of Testing Examples: {test_set_size}")



data=loadImages(img_path,mask=False)
true_mask=loadImages(mask_path,mask=True)

show_images(true_mask,3)

convert_to_binary_mask(true_mask)
show_images(true_mask,3)

true_mask=np.dot(true_mask[..., :3], [0.2989, 0.5870, 0.1140])
true_mask=np.expand_dims(true_mask, axis=-1)


show_images(true_mask,3)

true_mask[true_mask != 0.0]=1.0
data  = data / 255.0


show_images(true_mask,3)

newimgs = int(input('how many images do you want to create'))

augmentation(data,newimgs)

train_set_size = int(len(data) * 0.8)
print(f"Number of Training Examples: {train_set_size}")

validation_set_size = int(len(data) * 0.1)
print(f"Number of Validation Examples: {validation_set_size}")

test_set_size = len(data) - train_set_size - validation_set_size
print(f"Number of Testing Examples: {test_set_size}")
X_test=data[:test_set_size].transpose((0, 3, 1, 2))
y_test=true_mask[:test_set_size].transpose((0, 3, 1, 2))
print(f'test set size {X_test.shape}')


#rest of the data will be agumented and shuffled for training
data=data[test_set_size:]
true_mask=true_mask[test_set_size:]

# Shuffle the indices
shuffled_indices = np.random.permutation(len(data))

# Use the shuffled indices to shuffle both arrays
shuffled_data = data[shuffled_indices]
shuffled_masks = true_mask[shuffled_indices]
X_train=shuffled_data[:train_set_size].transpose((0, 3, 1, 2))
y_train=shuffled_masks[:train_set_size].transpose((0, 3, 1, 2))
print(f'training set size {X_train.shape}')

X_val=shuffled_data[train_set_size:train_set_size+validation_set_size].transpose((0, 3, 1, 2))
y_val=shuffled_masks[train_set_size:train_set_size+validation_set_size].transpose((0, 3, 1, 2))
print(f'val set size {X_val.shape}')


#create dataset class

train_dataset = ImageDataset(X_train,y_train)
val_dataset = ImageDataset(X_val, y_val)
test_dataset = ImageDataset(X_test, y_test)

# Create a DataLoader from the dataset
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader=DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader=DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


EPOCHS=EPOCHS
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

loss_fn = IoULoss().to(DEVICE)
# loss_fn =nn.BCELOSS().to(DEVICE)
LEARNING_RATE=LEARNING_RATE

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


loss=math.inf
os.makedirs('pretraining/pretraincheckpoints', exist_ok=True)

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

torch.cuda.empty_cache()

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, DEVICE)
    val_loss = eval_model(model, val_dataloader, loss_fn, DEVICE)
    
    if epoch % 5 == 0:
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        print(f'Train loss {train_loss}')
        print(f'Val   loss {val_loss}')
        print()
        
        # Save checkpoint every 5 epochs
        checkpoint_path = f'pretraincheckpoints/model_checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), os.path.join(BASE_DIR,'pretraining','best_model.pth'))
        best_val_loss = val_loss
        print(f"New best model saved with validation loss: {best_val_loss}")
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

print("Training completed.")

model.load_state_dict(torch.load(os.path.join(BASE_DIR,'pretraining','best_model.pth')))


plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='validation loss')

# plt.title('Training history')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend()




prediction = model(torch.tensor(X_test[29][np.newaxis,...], dtype=torch.float32, device=DEVICE)).cpu().detach().numpy()


#test cases
# Show the original and augmented images side by side
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(X_test[29].transpose(1,2,0))
ax[0].set_title("Original")

ax[1].imshow(y_test[29].transpose(1,2,0))
ax[1].set_title("original mask")


ax[2].imshow(prediction[0].transpose(1,2,0),cmap='gray')
ax[2].set_title("predicted mask")

plt.show()
