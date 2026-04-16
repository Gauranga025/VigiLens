import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/saved_model.keras")

# Load video (change path if needed)
cap = cv2.VideoCapture("Data/Test/test_video.avi")

# Check if video opened
if not cap.isOpened():
    print("❌ Error: Could not open video")
    exit()

frames = []
THRESHOLD = 0.05  # Based on your observed values

while True:
    ret, frame = cap.read()

    if not ret:
        print("✅ End of video")
        break

    # Preprocess frame
    frame_resized = cv2.resize(frame, (227, 227))
    gray = np.mean(frame_resized / 255.0, axis=2)

    frames.append(gray)

    # Process every 10 frames
    if len(frames) == 10:
        input_seq = np.array(frames)
        input_seq = input_seq.reshape(1, 227, 227, 10, 1)

        # Predict
        output = model.predict(input_seq, verbose=0)

        # Compute reconstruction error
        error = np.mean((input_seq - output) ** 2)

        # Decision
        if error > THRESHOLD:
            label = "ANOMALY"
            color = (0, 0, 255)  # Red
        else:
            label = "NORMAL"
            color = (0, 255, 0)  # Green

        # Print error
        print(f"Error: {error:.5f} → {label}")

        # Display result
        display_frame = frame.copy()
        cv2.putText(display_frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(display_frame, f"Error: {error:.4f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Anomaly Detection", display_frame)

        # Reset frames buffer
        frames = []

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()