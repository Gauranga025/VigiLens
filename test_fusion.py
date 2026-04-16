import cv2
import numpy as np
from collections import deque

from models.fusion_model import build_fusion_model
from models.anomaly_model import load_anomaly_model
from utils.preprocess import preprocess_visible, preprocess_ir
from utils.segmentation import segment_objects

# Load models
fusion_model = build_fusion_model()
anomaly_model = load_anomaly_model()

sequence = deque(maxlen=10)

cap = cv2.VideoCapture("Data/Test/video.avi")

THRESHOLD = 0.04

print("System started...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Simulate IR (for now use grayscale)
    ir_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ir_frame = np.expand_dims(ir_frame, axis=-1)

    # Preprocess
    vis = preprocess_visible(frame)
    ir = preprocess_ir(frame)

    vis = np.expand_dims(vis, axis=0)
    ir = np.expand_dims(ir, axis=0)

    # Fusion
    fused = fusion_model.predict([vis, ir], verbose=0)[0]

    # Convert fused → grayscale for ConvLSTM
    fused_gray = np.mean(fused, axis=2)

    sequence.append(fused_gray)

    # Segmentation
    seg_frame, boxes = segment_objects(frame)

    if len(sequence) == 10:
        input_seq = np.array(sequence)
        input_seq = input_seq.reshape(1, 227, 227, 10, 1)

        output = anomaly_model.predict(input_seq, verbose=0)

        error = np.mean((input_seq - output)**2)

        print(f"Error: {error:.5f}")

        if error > THRESHOLD:
            cv2.putText(seg_frame, "ANOMALY", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Fusion + Segmentation + Anomaly", seg_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()