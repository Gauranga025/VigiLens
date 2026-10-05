"""
VigiLens Streamlit Application - Multimodal Anomaly Detection

This application provides a web interface for the VigiLens multimodal
anomaly detection system using visible and IR/thermal video inputs.

Architecture:
- YOLO segmentation for pedestrian ROI
- ResNet50 feature extraction (RGB + IR)
- Modality-aware fusion (supports missing modalities)
- Causal streaming LSTM for temporal modeling
- Next-feature prediction for anomaly detection
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
from collections import deque
import json

from config.config import SystemConfig, get_config
from pipeline.frame_source import create_frame_source
from pipeline.inference import InferencePipeline
from models.anomaly_model import VigiLensModel
from models.segmentation import YOLOSegmentation
from training.calibration import AnomalyCalibrator

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

# Processing settings
st.sidebar.subheader("Processing")
device = st.sidebar.selectbox("Device", ["cuda", "cpu"])

# Model settings
st.sidebar.subheader("Model")
checkpoint_path = st.sidebar.text_input("Checkpoint Path", "checkpoints/best_model.pth")
calibration_path = st.sidebar.text_input("Calibration Path", "checkpoints/calibration.json")

# Temporal smoothing
st.sidebar.subheader("Temporal Smoothing")
smoothing_method = st.sidebar.selectbox("Smoothing Method", ["moving_average", "exponential"])
window_size = st.sidebar.slider("Window Size", 1, 30, 10)

# Display settings
st.sidebar.subheader("Display")
show_ir = st.sidebar.checkbox("Show IR Frame", value=True)
show_mask = st.sidebar.checkbox("Show Segmentation Mask", value=False)

st.sidebar.markdown("---")
st.sidebar.text("Model: ResNet50 + LSTM")
st.sidebar.text("Method: Next-feature prediction")
st.sidebar.text("Training: Normal-only")
st.sidebar.text("Fusion: Modality-aware")

# ------------------ FILE INPUT ------------------
st.subheader("Input Sources")

col_vis, col_ir = st.columns(2)

with col_vis:
    visible_file = st.file_uploader("Visible Video", type=["mp4", "avi", "mov"], key="visible")

with col_ir:
    ir_file = st.file_uploader("IR/Thermal Video (Optional)", type=["mp4", "avi", "mov"], key="ir")

# ------------------ MAIN DASHBOARD ------------------
if visible_file:
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
            # Load configuration
            config = get_config()
            config.model.device = device

            # Load model
            model = VigiLensModel(device=device)
            if Path(checkpoint_path).exists():
                model.load_checkpoint(checkpoint_path)
            else:
                st.warning(f"Checkpoint not found: {checkpoint_path}. Using untrained model.")

            # Load calibration
            calibration_info = None
            if Path(calibration_path).exists():
                calibration_info = AnomalyCalibrator.load_calibration(calibration_path)
                st.info(f"Calibration loaded: threshold={calibration_info.get('threshold_std', 0.5):.4f}")
            else:
                st.warning(f"Calibration not found: {calibration_path}. Using default threshold.")

            # Initialize segmentation
            segmentation = YOLOSegmentation(device=device)

            # Initialize inference pipeline
            pipeline = InferencePipeline(
                model=model,
                segmentation=segmentation,
                calibration_info=calibration_info,
                smoothing_window=window_size,
                smoothing_method=smoothing_method
            )

            # Load frame source
            frame_source = create_frame_source(visible_path.name, ir_path.name if ir_path else None)

        st.success("Pipeline initialized successfully")

        # Main display columns
        col_video, col_metrics = st.columns([3, 1])

        with col_video:
            video_placeholder = st.empty()
            ir_placeholder = st.empty() if show_ir and ir_path else None
            mask_placeholder = st.empty() if show_mask else None

        with col_metrics:
            st.subheader("Metrics")
            m_score = st.empty()
            m_status = st.empty()
            m_fps = st.empty()
            m_frame = st.empty()
            m_ir = st.empty()

            st.markdown("---")
            st.subheader("Info")
            m_threshold = st.empty()

        status_box = st.empty()

        # Process video
        frame_count = 0
        score_history = deque(maxlen=50)

        while True:
            visible_frame, ir_frame = frame_source.read()

            if visible_frame is None:
                break

            # Process frame
            result = pipeline.process_frame(visible_frame, ir_frame)

            # Get segmentation mask for visualization
            mask = segmentation.extract_person_mask(visible_frame)

            # Draw anomaly indicator
            if result['is_anomalous']:
                cv2.rectangle(visible_frame, (0, 0), (visible_frame.shape[1], visible_frame.shape[0]), (0, 0, 255), 4)
                cv2.putText(visible_frame, "ANOMALY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(visible_frame, f"Score: {result['smoothed_score']:.3f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Update metrics
            m_score.metric("Anomaly Score", f"{result['smoothed_score']:.4f}")
            m_status.metric("Status", "ANOMALY" if result['is_anomalous'] else "NORMAL")
            m_fps.metric("FPS", f"{1.0/result['inference_time']:.1f}")
            m_frame.metric("Frame", frame_count)
            m_ir.metric("IR Available", "Yes" if result['ir_available'] else "No")

            if calibration_info:
                threshold = calibration_info.get('threshold_std', 0.5)
                m_threshold.metric("Threshold", f"{threshold:.4f}")

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

            if mask_placeholder and mask is not None:
                mask_display = cv2.resize(mask, (720, 480))
                mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2RGB)
                mask_placeholder.image(mask_display, channels="RGB")

            frame_count += 1
            score_history.append(result['smoothed_score'])

        # Cleanup
        frame_source.release()
        pipeline.reset_stream()
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