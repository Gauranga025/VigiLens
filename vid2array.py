import cv2
import numpy as np
import os
import shutil

train_path = "Data/Train"
frames_path = os.path.join(train_path, "frames")

if os.path.exists(frames_path):
    shutil.rmtree(frames_path)

os.makedirs(frames_path)

frame_count = 0

for video in os.listdir(train_path):
    if video.endswith(".avi") or video.endswith(".mp4"):
        cap = cv2.VideoCapture(os.path.join(train_path, video))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (227, 227))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray / 255.0

            cv2.imwrite(f"{frames_path}/{frame_count}.jpg", gray * 255)
            frame_count += 1

        cap.release()

print("Frames:", frame_count)

image_data = []

for img in sorted(os.listdir(frames_path), key=lambda x: int(x.split('.')[0])):
    path = os.path.join(frames_path, img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    image_data.append(img)

image_data = np.array(image_data)

frames = image_data.shape[0]
frames = frames - frames % 10
image_data = image_data[:frames]

image_data = image_data.reshape(-1, 227, 227, 10)
image_data = np.expand_dims(image_data, axis=4)

np.save("training.npy", image_data)

print("Saved training.npy:", image_data.shape)