import numpy as np
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import from config.py in the main directory
from config import * 





# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

original_matrix = np.array([[458.62594473, 0., 332.48800248],
                            [0., 451.44030872, 229.51157475],
                            [0., 0., 1.]])
dist_coeffs = np.array([0.38967341, -0.83674074, 0.32281102, 0.05279317, 1.66451797])
known_road_width = 10  # meters
camera_height = 1.55  # meters
camera_pitch = 12  # degrees (positive is tilted downwards)




