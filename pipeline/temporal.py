"""
Temporal smoothing module for anomaly scores.

This module implements temporal smoothing to reduce noise in anomaly detection
by considering scores across multiple frames.
"""

import numpy as np
from typing import Literal, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    Temporal smoother for anomaly scores.
    
    Supports multiple smoothing strategies:
    - moving_average: Simple moving average over window
    - exponential: Exponential moving average
    - consecutive: Requires consecutive anomalous frames
    """
    
    def __init__(self,
                 method: Literal["moving_average", "exponential", "consecutive"] = "moving_average",
                 window_size: int = 10,
                 consecutive_frames: int = 3,
                 alpha: float = 0.3):
        """
        Initialize temporal smoother.
        
        Args:
            method: Smoothing method
            window_size: Window size for moving average
            consecutive_frames: Required consecutive anomalous frames
            alpha: Smoothing factor for exponential smoothing
        """
        self.method = method
        self.window_size = window_size
        self.consecutive_frames = consecutive_frames
        self.alpha = alpha
        
        # History buffers
        self.score_history: Deque[float] = deque(maxlen=window_size)
        self.anomaly_history: Deque[bool] = deque(maxlen=window_size)
        
        # Exponential smoothing state
        self.ema_score = None
        
        logger.info(f"Temporal smoother: method={method}, window={window_size}")
    
    def smooth(self, score: float, is_anomalous: bool) -> tuple[float, bool]:
        """
        Apply temporal smoothing to anomaly score.
        
        Args:
            score: Raw anomaly score
            is_anomalous: Raw anomaly decision
        
        Returns:
            Tuple of (smoothed_score, smoothed_is_anomalous)
        """
        if self.method == "moving_average":
            return self._moving_average(score, is_anomalous)
        elif self.method == "exponential":
            return self._exponential_smoothing(score, is_anomalous)
        elif self.method == "consecutive":
            return self._consecutive_frames(score, is_anomalous)
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")
    
    def _moving_average(self, score: float, is_anomalous: bool) -> tuple[float, bool]:
        """Moving average smoothing."""
        self.score_history.append(score)
        self.anomaly_history.append(is_anomalous)
        
        if len(self.score_history) < self.window_size:
            # Not enough history yet
            smoothed_score = score
            smoothed_anomalous = is_anomalous
        else:
            smoothed_score = np.mean(self.score_history)
            # Require majority of frames to be anomalous
            smoothed_anomalous = sum(self.anomaly_history) > (self.window_size / 2)
        
        return smoothed_score, smoothed_anomalous
    
    def _exponential_smoothing(self, score: float, is_anomalous: bool) -> tuple[float, bool]:
        """Exponential moving average smoothing."""
        if self.ema_score is None:
            self.ema_score = score
        else:
            self.ema_score = self.alpha * score + (1 - self.alpha) * self.ema_score
        
        # For anomaly decision, use simple threshold on smoothed score
        # This assumes threshold is applied externally
        return self.ema_score, is_anomalous
    
    def _consecutive_frames(self, score: float, is_anomalous: bool) -> tuple[float, bool]:
        """Consecutive frames rule."""
        self.score_history.append(score)
        self.anomaly_history.append(is_anomalous)
        
        if len(self.anomaly_history) < self.consecutive_frames:
            smoothed_anomalous = False
        else:
            # Check if last N frames are all anomalous
            recent = list(self.anomaly_history)[-self.consecutive_frames:]
            smoothed_anomalous = all(recent)
        
        smoothed_score = np.mean(self.score_history) if self.score_history else score
        return smoothed_score, smoothed_anomalous
    
    def reset(self):
        """Reset smoother state."""
        self.score_history.clear()
        self.anomaly_history.clear()
        self.ema_score = None
        logger.info("Temporal smoother reset")
