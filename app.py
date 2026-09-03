"""
VigiLens Streamlit Application - Multimodal Anomaly Detection

This application provides a web interface for the VigiLens multimodal
anomaly detection system using visible and IR/thermal video inputs.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
from collections import deque

from config.config import SystemConfig, get_config
from pipeline.multimodal_pipeline import MultimodalAnomalyPipeline
from ultralytics import YOLO

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
col_title, col_status = st.columns([6, 1])

with col_title:
    st.markdown("### VigiLens")
    st.caption("Multimodal Visible + IR Anomaly Detection System")

with col_status:
    st.success("Active")

st.markdown("---")

# ------------------ SIDEBAR CONFIGURATION ------------------
st.sidebar.header("System Configuration")

# Anomaly detection settings
st.sidebar.subheader("Anomaly Detection")
anomaly_threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.65, 0.05)
distance_metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "cosine"])
adaptive_threshold = st.sidebar.checkbox("Adaptive Threshold", value=False)

# Temporal smoothing
st.sidebar.subheader("Temporal Smoothing")
smoothing_method = st.sidebar.selectbox("Smoothing Method", ["moving_average", "exponential", "consecutive"])
window_size = st.sidebar.slider("Window Size", 1, 30, 10)
consecutive_frames = st.sidebar.slider("Consecutive Frames", 1, 10, 3)

# Fusion settings
st.sidebar.subheader("Multimodal Fusion")
fusion_method = st.sidebar.selectbox("Fusion Method", ["concat", "weighted", "average"])
visible_weight = st.sidebar.slider("Visible Weight", 0.0, 1.0, 0.6, 0.1)

# Display settings
st.sidebar.subheader("Display")
show_ir = st.sidebar.checkbox("Show IR Frame", value=True)
show_bboxes = st.sidebar.checkbox("Show Object Bounding Boxes", value=True)

st.sidebar.markdown("---")
st.sidebar.text("Model: ResNet50 (Pretrained)")
st.sidebar.text("Method: Distance-based Scoring")
st.sidebar.text("No training required")

# ------------------ FILE INPUT ------------------
st.subheader("Input Sources")

col_vis, col_ir = st.columns(2)

with col_vis:
    visible_file = st.file_uploader("Visible Video", type=["mp4", "avi", "mov"], key="visible")

with col_ir:
    ir_file = st.file_uploader("IR/Thermal Video (Optional)", type=["mp4", "avi", "mov"], key="ir")

# ------------------ MAIN DASHBOARD ------------------
if visible_file:
    # Update config
    config = get_config()
    config.anomaly.anomaly_threshold = anomaly_threshold
    config.anomaly.distance_metric = distance_metric
    config.anomaly.adaptive_threshold = adaptive_threshold
    config.temporal.smoothing_method = smoothing_method
    config.temporal.window_size = window_size
    config.temporal.consecutive_frames = consecutive_frames
    config.fusion.fusion_method = fusion_method
    config.fusion.visible_weight = visible_weight
    config.fusion.ir_weight = 1.0 - visible_weight

    # Save uploaded files
    visible_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    visible_path.write(visible_file.read())

    ir_path = None
    if ir_file:
        ir_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        ir_path.write(ir_file.read())
        st.info("IR video loaded - multimodal mode")
    else:
        st.warning("IR video not provided - visible only mode")

    # Initialize pipeline
    try:
        with st.spinner("Initializing models..."):
            pipeline = MultimodalAnomalyPipeline(config)
            pipeline.load_source(visible_path.name, ir_path.name if ir_path else None)

            # Load YOLO for object detection (auxiliary)
            yolo_model = YOLO("yolov8n.pt")

        st.success("Pipeline initialized successfully")

        # Main display columns
        col_video, col_metrics = st.columns([3, 1])

        with col_video:
            video_placeholder = st.empty()
            ir_placeholder = st.empty() if show_ir and ir_path else None

        with col_metrics:
            st.subheader("Metrics")
            m_score = st.empty()
            m_status = st.empty()
            m_fps = st.empty()
            m_frame = st.empty()
            m_ir = st.empty()

            st.markdown("---")
            st.subheader("Objects")
            m_objects = st.empty()

        status_box = st.empty()

        # Process video
        frame_count = 0
        score_history = deque(maxlen=50)

        while True:
            visible_frame, ir_frame = pipeline.frame_source.read()

            if visible_frame is None:
                break

            # Process frame
            result = pipeline.process_frame(visible_frame, ir_frame)

            # YOLO detection (auxiliary)
            yolo_results = yolo_model(visible_frame)
            object_count = len(yolo_results[0].boxes)

            # Draw bounding boxes if enabled
            if show_bboxes:
                for box in yolo_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(visible_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw anomaly indicator
            if result['is_anomalous']:
                cv2.rectangle(visible_frame, (0, 0), (visible_frame.shape[1], visible_frame.shape[0]), (0, 0, 255), 4)
                cv2.putText(visible_frame, "ANOMALY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(visible_frame, f"Score: {result['smoothed_score']:.3f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Update metrics
            m_score.metric("Anomaly Score", f"{result['smoothed_score']:.3f}")
            m_status.metric("Status", "ANOMALY" if result['is_anomalous'] else "NORMAL")
            m_fps.metric("FPS", f"{1.0/result['inference_time']:.1f}")
            m_frame.metric("Frame", frame_count)
            m_ir.metric("IR Available", "Yes" if result['ir_available'] else "No")
            m_objects.metric("Objects", object_count)

            # Update status box
            if result['is_anomalous']:
                status_box.error("ANOMALY DETECTED")
            else:
                status_box.success("NORMAL")

            # Display frames
            visible_display = cv2.resize(visible_frame, (720, 480))
            video_placeholder.image(visible_display, channels="BGR")

            if ir_placeholder and ir_frame is not None:
                ir_display = cv2.resize(ir_frame, (720, 480))
                if len(ir_display.shape) == 2:
                    ir_display = cv2.cvtColor(ir_display, cv2.COLOR_GRAY2RGB)
                ir_placeholder.image(ir_display, channels="RGB")

            frame_count += 1
            score_history.append(result['smoothed_score'])

        # Cleanup
        pipeline.frame_source.release()
        Path(visible_path.name).unlink(missing_ok=True)
        if ir_path:
            Path(ir_path.name).unlink(missing_ok=True)

        st.success(f"Processing complete - {frame_count} frames processed")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        if visible_path:
            Path(visible_path.name).unlink(missing_ok=True)
        if ir_path:
            Path(ir_path.name).unlink(missing_ok=True)

else:
    st.info("Please upload a visible video to begin analysis")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("VigiLens | Multimodal Anomaly Detection | NIT Rourkela")