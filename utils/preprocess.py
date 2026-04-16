import cv2
import numpy as np

def preprocess_visible(frame):
    frame = cv2.resize(frame, (227, 227))
    frame = frame / 255.0
    return frame

def preprocess_ir(frame):
    frame = cv2.resize(frame, (227, 227))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    gray = np.expand_dims(gray, axis=-1)
    return gray