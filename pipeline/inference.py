"""
Stateful real-time inference pipeline for VigiLens.

Handles streaming inference with:
- LSTM hidden state persistence
- One-frame prediction latency
- Temporal anomaly smoothing
- Calibration-based thresholding
"""

import torch
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from collections import deque
import logging
import time

from models.anomaly_model import VigiLensModel
from models.segmentation import YOLOSegmentation
from pipeline.frame_source import MultimodalFrameSource
from training.calibration import AnomalyCalibrator

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Stateful inference pipeline for real-time anomaly detection.

    Maintains LSTM hidden state across frames for streaming inference.
    Implements one-frame prediction latency for causal behavior.
    """

    def __init__(self,
                 model: VigiLensModel,
                 segmentation: YOLOSegmentation,
                 calibration_info: Optional[Dict] = None,
                 smoothing_window: int = 10,
                 smoothing_method: str = "moving_average"):
        """
        Initialize inference pipeline.

        Args:
            model: Trained VigiLens model
            segmentation: YOLO segmentation model
            calibration_info: Calibration statistics (threshold, etc.)
            smoothing_window: Window size for temporal smoothing
            smoothing_method: Smoothing method ('moving_average', 'exponential')
        """
        self.model = model
        self.segmentation = segmentation
        self.calibration_info = calibration_info or {}
        self.smoothing_window = smoothing_window
        self.smoothing_method = smoothing_method

        # Get threshold from calibration
        self.threshold = self.calibration_info.get('threshold_std', 0.5)

        # Temporal smoothing state
        self.score_history: deque = deque(maxlen=smoothing_window)
        self.ema_score: Optional[float] = None
        self.alpha: float = 0.3  # EMA smoothing factor

        # Statistics
        self.frame_count = 0
        self.total_inference_time = 0.0

        logger.info(f"Inference pipeline initialized: threshold={self.threshold:.4f}")

    def reset_stream(self):
        """Reset streaming state (LSTM hidden state, smoothing, counters)."""
        self.model.reset_stream()
        self.score_history.clear()
        self.ema_score = None
        self.frame_count = 0
        self.total_inference_time = 0.0
        logger.info("Inference stream reset")

    def process_frame(self,
                     rgb_frame: np.ndarray,
                     ir_frame: Optional[np.ndarray] = None) -> Dict:
        """
        Process a single frame with stateful inference.

        Args:
            rgb_frame: RGB frame (H, W, 3)
            ir_frame: IR frame (H, W) or None

        Returns:
            Dictionary containing:
            - raw_score: Prediction error
            - smoothed_score: Temporally smoothed score
            - is_anomalous: Boolean anomaly decision
            - rgb_available: bool
            - ir_available: bool
            - inference_time: Processing time
        """
        start_time = time.time()

        # Determine modality availability
        rgb_available = rgb_frame is not None
        ir_available = ir_frame is not None

        # Get segmentation mask based on available modalities
        if rgb_available:
            # RGB + IR or RGB only: Use YOLO segmentation on RGB
            rgb_mask = self.segmentation.extract_person_mask(rgb_frame)
        else:
            # IR only: No RGB segmentation available, use None (full frame)
            rgb_mask = None

        # Handle IR mask
        if ir_available and rgb_available and rgb_mask is not None:
            # RGB + IR: Resize RGB mask to IR dimensions
            ir_mask = cv2.resize(rgb_mask, (ir_frame.shape[1], ir_frame.shape[0]))
        elif ir_available and not rgb_available:
            # IR only: No RGB mask, use None (full IR frame)
            ir_mask = None
        else:
            # RGB only: IR mask not applicable
            ir_mask = None

        # Process frame through model
        result = self.model.process_frame(
            rgb_frame,
            ir_frame,
            rgb_mask,
            ir_mask,
            rgb_available,
            ir_available
        )

        raw_score = result['anomaly_score'] if result['anomaly_score'] is not None else 0.0

        # Temporal smoothing
        smoothed_score = self._smooth_score(raw_score)

        # Anomaly decision
        is_anomalous = smoothed_score > self.threshold

        # Update statistics
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1

        return {
            'raw_score': raw_score,
            'smoothed_score': smoothed_score,
            'is_anomalous': is_anomalous,
            'rgb_available': rgb_available,
            'ir_available': ir_available,
            'inference_time': inference_time
        }

    def _smooth_score(self, score: float) -> float:
        """
        Apply temporal smoothing to anomaly score.

        Args:
            score: Current raw anomaly score

        Returns:
            Smoothed anomaly score
        """
        if self.smoothing_method == "moving_average":
            self.score_history.append(score)
            if len(self.score_history) < self.smoothing_window:
                return score
            return np.mean(self.score_history)

        elif self.smoothing_method == "exponential":
            if self.ema_score is None:
                self.ema_score = score
            else:
                self.ema_score = self.alpha * score + (1 - self.alpha) * self.ema_score
            return self.ema_score

        else:
            # No smoothing
            return score

    def process_video(self,
                     frame_source: MultimodalFrameSource) -> Dict:
        """
        Process entire video from frame source.

        Args:
            frame_source: Frame source for RGB/IR video

        Returns:
            Dictionary containing overall statistics
        """
        self.reset_stream()

        results = []
        anomaly_count = 0

        while True:
            rgb_frame, ir_frame = frame_source.read()

            if rgb_frame is None:
                break

            result = self.process_frame(rgb_frame, ir_frame)
            results.append(result)

            if result['is_anomalous']:
                anomaly_count += 1

        # Compute statistics
        avg_inference_time = self.total_inference_time / self.frame_count if self.frame_count > 0 else 0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        anomaly_rate = anomaly_count / self.frame_count if self.frame_count > 0 else 0

        return {
            'total_frames': self.frame_count,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'results': results
        }

    def get_fps(self) -> float:
        """Get current average FPS."""
        if self.total_inference_time > 0 and self.frame_count > 0:
            return self.frame_count / self.total_inference_time
        return 0.0
