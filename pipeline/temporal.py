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
    Temporal smoother for anomaly scores with explicit decision logic.
    
    Separates three stages:
    1. Frame-level anomaly score (input)
    2. Temporal aggregation (smoothing)
    3. Final anomaly decision (thresholding)
    
    Supports multiple smoothing strategies:
    - moving_average: Simple moving average over window
    - exponential: Exponential moving average
    - consecutive: Requires consecutive anomalous frames
    """
    
    def __init__(self,
                 method: Literal["moving_average", "exponential", "consecutive"] = "moving_average",
                 window_size: int = 10,
                 consecutive_frames: int = 3,
                 alpha: float = 0.3,
                 decision_threshold: float = 0.5):
        """
        Initialize temporal smoother.
        
        Args:
            method: Smoothing method
            window_size: Window size for moving average
            consecutive_frames: Required consecutive anomalous frames
            alpha: Smoothing factor for exponential smoothing
            decision_threshold: Threshold for final anomaly decision
        """
        self.method = method
        self.window_size = window_size
        self.consecutive_frames = consecutive_frames
        self.alpha = alpha
        self.decision_threshold = decision_threshold
        
        # History buffers
        self.score_history: Deque[float] = deque(maxlen=window_size)
        self.anomaly_history: Deque[bool] = deque(maxlen=window_size)
        
        # Exponential smoothing state
        self.ema_score = None
        
        logger.info(f"Temporal smoother: method={method}, window={window_size}, threshold={decision_threshold}")
    
    def smooth(self, score: float, is_anomalous: bool) -> tuple[float, bool]:
        """
        Apply temporal smoothing to anomaly score.
        
        This performs temporal aggregation. The final decision is made
        by applying the decision_threshold to the smoothed score.
        
        Args:
            score: Frame-level anomaly score
            is_anomalous: Frame-level anomaly decision (may be ignored)
        
        Returns:
            Tuple of (smoothed_score, final_anomaly_decision)
        """
        if self.method == "moving_average":
            smoothed_score = self._moving_average(score)
        elif self.method == "exponential":
            smoothed_score = self._exponential_smoothing(score)
        elif self.method == "consecutive":
            smoothed_score, final_decision = self._consecutive_frames(score, is_anomalous)
            return smoothed_score, final_decision
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")
        
        # Apply threshold for final decision
        final_decision = smoothed_score > self.decision_threshold
        return smoothed_score, final_decision
    
    def _moving_average(self, score: float) -> float:
        """Moving average smoothing."""
        self.score_history.append(score)
        
        if len(self.score_history) < self.window_size:
            # Not enough history yet, return current score
            return score
        else:
            return np.mean(self.score_history)
    
    def _exponential_smoothing(self, score: float) -> float:
        """Exponential moving average smoothing."""
        if self.ema_score is None:
            self.ema_score = score
        else:
            self.ema_score = self.alpha * score + (1 - self.alpha) * self.ema_score
        return self.ema_score
    
    def _consecutive_frames(self, score: float, is_anomalous: bool) -> tuple[float, bool]:
        """Consecutive frames rule - requires N consecutive anomalous frames."""
        self.score_history.append(score)
        self.anomaly_history.append(is_anomalous)
        
        if len(self.anomaly_history) < self.consecutive_frames:
            # Not enough history yet
            final_decision = False
        else:
            # Check if last N frames are all anomalous
            recent = list(self.anomaly_history)[-self.consecutive_frames:]
            final_decision = all(recent)
        
        smoothed_score = np.mean(self.score_history) if self.score_history else score
        return smoothed_score, final_decision
    
    def reset(self):
        """Reset smoother state."""
        self.score_history.clear()
        self.anomaly_history.clear()
        self.ema_score = None
        logger.info("Temporal smoother reset")
