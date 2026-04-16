import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import tempfile

# ------------------ CONFIG ------------------
st.set_page_config(page_title="VigiLens", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}
h1, h2, h3 {
    color: #ffffff;
}
.metric-box {
    padding: 10px;
    border-radius: 8px;
    background-color: #1c1f26;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
col_title, col_status = st.columns([6,1])

with col_title:
    st.markdown("### VigiLens")
    st.caption("AI-Powered Activity Anomaly Detection System")

with col_status:
    st.success("Active")

st.markdown("---")

# ------------------ SIDEBAR ------------------
st.sidebar.header("System Configuration")

threshold = st.sidebar.slider("Anomaly Sensitivity", 0.2, 0.7, 0.4)

st.sidebar.markdown("---")
st.sidebar.text("Model: YOLOv8")
st.sidebar.text("Mode: Real-Time Monitoring")

# ------------------ LOAD MODEL ------------------
model = YOLO("yolov8n.pt")

# ------------------ MAIN DASHBOARD ------------------
left, right = st.columns([3,1])

video_placeholder = left.empty()

with right:
    st.subheader("System Metrics")

    m1 = st.empty()
    m2 = st.empty()
    m3 = st.empty()

    st.markdown("---")
    st.subheader("Activity Trend")
    chart = st.line_chart([])

status_box = st.empty()

# ------------------ ALERT FUNCTION ------------------
def generate_alert(drop):
    if drop > 0.6:
        return "CRITICAL"
    elif drop > 0.4:
        return "WARNING"
    return "NORMAL"

# ------------------ FILE INPUT ------------------
uploaded_file = st.file_uploader("Upload Surveillance Video", type=["mp4","avi","mov"])

if uploaded_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    count_history = deque(maxlen=20)
    alert_buffer = deque(maxlen=5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (720, 480))

        # YOLO Detection
        results = model(frame)
        detections = results[0].boxes
        object_count = len(detections)

        count_history.append(object_count)

        # Draw bounding boxes
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        if len(count_history) == 20:
            avg_count = np.mean(count_history)
            drop_ratio = (avg_count - object_count) / (avg_count + 1e-8)

            alert = drop_ratio > threshold
            alert_buffer.append(alert)

            # Metrics
            m1.metric("Objects", object_count)
            m2.metric("Average", int(avg_count))
            m3.metric("Drop", f"{drop_ratio:.2f}")

            status = generate_alert(drop_ratio)

            if sum(alert_buffer) >= 2:
                status_box.error(f"Status: {status}")
                cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 2)
            else:
                status_box.success("Status: NORMAL")

        video_placeholder.image(frame, channels="BGR")
        chart.add_rows([object_count])

    cap.release()
    st.success("Processing Complete")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("VigiLens | AI Surveillance System | NIT Rourkela")