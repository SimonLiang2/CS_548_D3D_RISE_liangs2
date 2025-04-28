import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

# tcp normalization and gripper width normalization
TRANS_MIN, TRANS_MAX = np.array([-0.7, -0.7, -0.2]), np.array([0.7, 0.7, 1.2]) # Expanded by 0.2 meters
MAX_GRIPPER_WIDTH = 0.11 # meter

# workspace in camera coordinate
WORKSPACE_MIN = np.array([-0.7, -0.7, -0.2]) # Expanded by 0.2 meters
WORKSPACE_MAX = np.array([0.7, 0.7, 1.2]) # Expanded by 0.2 meters

# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0.0])
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.4])

# gripper threshold (to avoid gripper action too frequently)
GRIPPER_THRESHOLD = 0.02 # meter
