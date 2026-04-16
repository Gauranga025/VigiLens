import numpy as np
import os
import cv2

train_path = "Data/Train"
image_data = []

# Load frames
for folder in sorted(os.listdir(train_path)):
    folder_path = os.path.join(train_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".tif"):
            img_path = os.path.join(folder_path, file)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (227, 227))

            # Normalize
            img = img / 255.0

            # Convert to grayscale
            gray = np.mean(img, axis=2)

            image_data.append(gray)

image_data = np.array(image_data)

print("Total frames:", image_data.shape)

# Make sequences of 10 frames
frames = image_data.shape[0]
frames = frames - frames % 10

image_data = image_data[:frames]
image_data = image_data.reshape(-1, 227, 227, 10)
image_data = np.expand_dims(image_data, axis=4)

print("Final shape:", image_data.shape)

# Save
np.save("training.npy", image_data)

print("✅ Avenue_training.npy created")