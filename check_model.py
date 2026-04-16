import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

image_data = []

train_path = "Data/Train"
frames_path = os.path.join(train_path, "frames")

# Create frames folder
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

def data_store(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (227, 227))
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    image_data.append(gray)

# 🔥 STEP 1: Extract frames
count = 0
for video in os.listdir(train_path):
    if video.endswith((".mp4", ".avi")):
        video_path = os.path.join(train_path, video)
        cap = cv2.VideoCapture(video_path)

        success, frame = cap.read()
        while success:
            frame_path = os.path.join(frames_path, f"{count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            success, frame = cap.read()
            count += 1

        cap.release()

# 🔥 STEP 2: Load frames AFTER extraction
for image in os.listdir(frames_path):
    image_path = os.path.join(frames_path, image)
    data_store(image_path)

# 🔥 STEP 3: Convert to numpy
image_data = np.array(image_data)

print("Total frames:", image_data.shape)

# Normalize
image_data = (image_data - image_data.mean()) / (image_data.std() + 1e-8)
image_data = np.clip(image_data, 0, 1)

# Save
np.save("training.npy", image_data)

print("✅ training.npy created")