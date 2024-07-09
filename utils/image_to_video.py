import cv2
import os
import numpy as np
from glob import glob

def create_video_from_images(image_folder, output_video_path, fps=24):
    # Get list of image files
    images = sorted(glob(os.path.join(image_folder, '*.jpg')))  # Adjust file extension if needed
    
    if not images:
        print("No images found in the specified folder.")
        return
    
    # Read the first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' if mp4 doesn't work
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate through images and add to video
    for image_path in images:
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release the VideoWriter
    out.release()

    print(f"Video created successfully: {output_video_path}")

# Usage example
# create_video_from_images(image_folder, output_video, fps=30)