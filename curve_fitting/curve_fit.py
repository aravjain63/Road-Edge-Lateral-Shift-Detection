import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from matplotlib import pyplot as plt
from collections import defaultdict
import random
import os
from tqdm import tqdm
from curve_fitting.config import *

np.random.seed(42)
random.seed(42)

def estimate_distances_to_edges(left_x, right_x, camera_center_x, known_road_width, camera_matrix, 
                                camera_height, camera_pitch, y_target):
    # Calculate total pixel width of the road
    total_pixel_width = right_x - left_x
    
    # Calculate scale factor (meters per pixel)
    scale_factor = known_road_width / total_pixel_width
    
    # Calculate distances to left and right edges from car centerline
    left_distance = (camera_center_x - left_x) * scale_factor
    right_distance = (right_x - camera_center_x) * scale_factor
    
    # Calculate the focal length (average of fx and fy)
    focal_length = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2
    
    # Calculate the forward distance based on y_target
    forward_pixel_distance = abs(y_target - camera_matrix[1, 2])  # Distance from principal point to y_target
    forward_distance = (camera_height * focal_length) / forward_pixel_distance
    
    # Adjust forward distance based on camera pitch
    pitch_rad = np.radians(camera_pitch)
    adjusted_forward_distance = forward_distance * np.cos(pitch_rad) - camera_height * np.sin(pitch_rad)
    
    # Calculate Euclidean distances
    left_euclidean = np.sqrt(left_distance**2 + adjusted_forward_distance**2 + camera_height**2)
    right_euclidean = np.sqrt(right_distance**2 + adjusted_forward_distance**2 + camera_height**2)
    
    return left_distance, right_distance, adjusted_forward_distance, left_euclidean, right_euclidean

def detect_road_edges(binary_map):
    # Apply Canny edge detection
    height = binary_map.shape[0]

    # ignore bottom 20% for curve fitting
    y_threshold = int(0.8 * height)

    # Apply Canny edge detection
    edges = cv2.Canny(binary_map, 50, 150)
    # Create a mask to exclude vertical edges
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    edge_mask = edges - vertical_edges
    
    # Find coordinates of edge pixels, excluding the bottom 20%
    y_coords, x_coords = np.where((edge_mask == 255)& (np.arange(height)[:, np.newaxis] < y_threshold)) 
    
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError("No edge points found. Adjust Canny thresholds or check segmentation map.")
    
    edge_points = np.column_stack((x_coords, y_coords))
    
    # Create dictionaries to store min and max x-values for each y
    min_x_dict = defaultdict(lambda: float('inf'))
    max_x_dict = defaultdict(lambda: float('-inf'))
    y_counts = defaultdict(int)

    for x, y in edge_points:
        y_counts[y] += 1
        if y_counts[y] > 1:
            min_x_dict[y] = min(min_x_dict[y], x)
            max_x_dict[y] = max(max_x_dict[y], x)

    min_x_points = np.array([(min_x_dict[y], y) for y in min_x_dict])
    max_x_points = np.array([(max_x_dict[y], y) for y in max_x_dict])
    
    # Apply RANSAC on min_x_points (left edge) and max_x_points (right edge)
    ransac_min = RANSACRegressor(min_samples=3, max_trials=800, residual_threshold=4,random_state=42)
    ransac_max = RANSACRegressor(min_samples=3, max_trials=800, residual_threshold=4,random_state=42)
    
    ransac_min.fit(min_x_points[:, 1].reshape(-1, 1), min_x_points[:, 0])
    ransac_max.fit(max_x_points[:, 1].reshape(-1, 1), max_x_points[:, 0])
    
    return ransac_min, ransac_max, edge_mask

# Camera parameters
# original_matrix = np.array([[458.62594473, 0., 332.48800248],
#                             [0., 451.44030872, 229.51157475],
#                             [0., 0., 1.]])
# dist_coeffs = np.array([0.38967341, -0.83674074, 0.32281102, 0.05279317, 1.66451797])
# known_road_width = 10  # meters
# camera_height = 1.55  # meters
# camera_pitch = 12  # degrees (positive is tilted downwards)
