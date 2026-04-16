import cv2
import numpy as np
from collections import deque

from models.fusion_model import build_fusion_model
from models.anomaly_model import load_anomaly_model
from utils.preprocess import preprocess_visible, preprocess_ir
from utils.segmentation import segment_objects

fusion_model = build_fusion_model()
anomaly_model = load_anomaly_model()

sequence = deque(maxlen=10)

cap = cv2.VideoCapture("Data/Test/video.avi")

THRESHOLD = 0.04

print("Running system...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # 🔹 SEGMENTATION FIRST
    seg_frame, boxes = segment_objects(frame)

    # 🔹 PREPROCESS
    vis = preprocess_visible(seg_frame)
    ir = preprocess_ir(seg_frame)

    vis = np.expand_dims(vis, axis=0)
    ir = np.expand_dims(ir, axis=0)

    # 🔹 FUSION
    fused = fusion_model.predict([vis, ir], verbose=0)[0]

    fused_gray = np.mean(fused, axis=2)

    sequence.append(fused_gray)

    # 🔹 ANOMALY DETECTION
    if len(sequence) == 10:
        input_seq = np.array(sequence)
        input_seq = input_seq.reshape(1, 227, 227, 10, 1)

        output = anomaly_model.predict(input_seq, verbose=0)
        error = np.mean((input_seq - output) ** 2)

        print(f"Error: {error:.5f}")

        if error > THRESHOLD:
            cv2.rectangle(display, (0,0), (display.shape[1], display.shape[0]), (0,0,255), 3)
            cv2.putText(display, "ANOMALY", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("VigiLens System", display)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()