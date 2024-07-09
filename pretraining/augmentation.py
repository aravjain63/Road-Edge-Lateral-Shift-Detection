import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np

def augmentation(data,newimgs):
    #select number of images to be created
    NewIMGS=newimgs
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5,border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.RGBShift(r_shift_limit=0.5, g_shift_limit=0.5, b_shift_limit=0.5, p=0.5),

    ])
    data=data.astype(np.float32)
    test_aug=augmentation_pipeline(image=data[5])
    test_example=test_aug["image"]
    # Show the original and augmented images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(data[5])
    ax[0].set_title("Original")
    ax[1].imshow(test_example)
    ax[1].set_title("Augmented")
    plt.show()


    for i in range(len(data)):
        for j in range(NewIMGS):
            print(i)
            image = data[i]
            mask = true_mask[i]
            # Apply the augmentation pipeline
            augmented = augmentation_pipeline(image=image, mask=mask)

            image_augmented = augmented["image"]
            mask_augmented = augmented["mask"]
            data = np.vstack((data, image_augmented[np.newaxis,...]))
            true_mask = np.vstack((true_mask, mask_augmented[np.newaxis,...]))
