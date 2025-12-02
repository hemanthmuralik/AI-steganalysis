# In stego_encrypt.py
import cv2
import numpy as np

def calculate_complexity_map(img):
    # Calculate gradients (edges)
    # Data hidden in edges is harder to detect than data in smooth sky.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return magnitude

def encrypt_adaptive(image_path, message, output_path):
    img = cv2.imread(image_path)
    complexity = calculate_complexity_map(img)
    
    # Sort pixels by complexity (Highest complexity = Best place to hide)
    flat_indices = np.argsort(complexity.ravel())[::-1] # Descending order
    
    # ... logic to embed bits into these specific indices ...
    # This makes the attack 10x harder to detect!
