import numpy as np
from PIL import Image
import cv2

def image_transform(image_cv2):
    """
    Convert input image to downsampled binary version.
    input: cv2 image (np.array), shape (500, 500, 3)
    output: cv2 image (np.array), shape (32, 32)
    """
    # 1. Convert image to grayscale.
    image_cv2_grayscale = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    # 2. Resize. Uses bilinear interpolation by default.
    image_cv2_resized = cv2.resize(image_cv2_grayscale, (32, 32), cv2.INTER_LINEAR)
    # 3. Threshold image to 0-1.
    #_, image_cv2_threshold = cv2.threshold(image_cv2_resized, 0, 255, cv2.THRESH_BINARY)
    # 4. Normalize image between 0~1.
    image_normalized = image_cv2_resized / 255.
    return image_normalized

def lyapunov_measure():
    """
    Return lyapunov measure by creating a weighted matrix.
    """
    pixel_radius = 7
    measure = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            radius = np.linalg.norm(np.array([i - 15.5, j - 15.5]), ord=2) ** 2.0
            measure[i,j] = np.maximum(radius - pixel_radius, 0)
    return measure

def lyapunov(image_normalized):
    """
    Apply the lyapunov measure to the image. 
    input: cv2 image (np.array), shape (32, 32)
    output: (np.float), shape ()
    """
    V_measure = lyapunov_measure()
    # element-wise multiplication.
    V = np.sum(np.multiply(image_normalized, V_measure))
    # image_sum = np.sum(image_normalized)
    return V 
