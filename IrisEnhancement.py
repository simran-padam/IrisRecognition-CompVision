import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhacement(image):

    # Perform histogram equalization
    equalized = cv2.equalizeHist(image)


    return equalized