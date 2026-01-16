import cv2
import glob
import os
import random


def get_random_image(folder_path):
    """Selects a random image from the specified folder."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not files:
        return None
    return random.choice(files)

def resize_image(image, target_size=(448, 448)):
    """Resizes image to target size ensuring it fits within visual model constraints."""
    return cv2.resize(image, target_size)