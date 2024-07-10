
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from paths import *
from collections import defaultdict


def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()

    losses = []

    for train_input, train_mask in tqdm(data_loader):
        train_mask = train_mask.to(device)
        train_input=train_input.to(device)

        outputs=model(train_input.float())


        loss = loss_fn(outputs.float(), train_mask.float())

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()

    losses = []

    with torch.no_grad():
        for val_input, val_mask in data_loader:

            val_mask = val_mask.to(device)
            val_input=val_input.to(device)
            outputs=model(val_input.float())

            loss = loss_fn(outputs.float(), val_mask.float())
            losses.append(loss.item())

    return np.mean(losses)

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs=20, patience=5):
    history = defaultdict(list)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Create a unique directory for this run
    timestamp = int(time.time())
    run_name = f"run_{timestamp}"
    log_dir = os.path.join(BASE_DIR,'training','runs',run_name)
    checkpoint_dir = os.path.join(BASE_DIR,'training','checkpoints',run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize TensorBoard writer with the specific log directory
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_model(model, val_loader, loss_fn, device)

        # Log losses to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)

        if epoch % 5 == 0:
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            print(f'Train loss {train_loss}')
            print(f'Val   loss {val_loss}')
            print()

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(BASE_DIR,checkpoint_dir,'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"EarlyStopping counter: {early_stopping_counter} out of {patience}")

        if early_stopping_counter >= patience:
            print("Early stopping")
            break

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    # Close the TensorBoard writer
    writer.close()

    return history, log_dir, checkpoint_dir


def iou_score(y_pred, y_true):
    intersection = np.logical_and(y_pred, y_true).sum()
    union = np.logical_or(y_pred, y_true).sum()
    return intersection / (union + 1e-6)


def evaluate_model(model, test_dataset, device, batch_size=32, threshold=0.5):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_targets = []
    iou_scores = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            print(outputs.shape)
            preds = (outputs > threshold).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            
            # Calculate IoU for each image in the batch
            for pred, target in zip(preds, targets):
                iou_scores.append(iou_score(pred.cpu().numpy(), target.cpu().numpy()))
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')
    mean_iou = np.mean(iou_scores)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': mean_iou
    }


def visualize_predictions(model, test_dataset, device, num_samples=40):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            preds = (outputs > 0.75).float()
            
            # Move tensors to CPU and convert to numpy for visualization
            input_np = inputs[0].cpu().numpy().transpose(1, 2, 0)
            target_np = targets[0].cpu().numpy().squeeze()
            pred_np = preds[0].cpu().numpy().squeeze()
            
            axs[i, 0].imshow(input_np)
            axs[i, 0].set_title('Input Image')
            axs[i, 1].imshow(target_np, cmap='gray')
            axs[i, 1].set_title('Ground Truth')
            axs[i, 2].imshow(pred_np, cmap='gray')
            axs[i, 2].set_title('Prediction')
    
    plt.tight_layout()
    plt.show()