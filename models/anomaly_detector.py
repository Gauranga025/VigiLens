"""
Anomaly detection module using distance-based scoring.

This module implements anomaly detection using pretrained feature extraction
and distance-based anomaly scoring. No training is required.

Method:
1. Extract features using pretrained encoders
2. Compute distance from reference (normal) features
3. Apply threshold to detect anomalies

This is a simple but effective approach that:
- Requires no project-specific training
- Works with pretrained features
- Is explainable and academically defensible
"""

import numpy as np
from typing import Literal, Optional, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DistanceAnomalyDetector:
    """
    Distance-based anomaly detector.
    
    Computes anomaly score as distance from reference features.
    Uses a running average of recent features as the reference.
    """
    
    def __init__(self,
                 distance_metric: Literal["euclidean", "cosine"] = "euclidean",
                 threshold: float = 0.65,
                 adaptive: bool = False,
                 window_size: int = 100):
        """
        Initialize anomaly detector.
        
        Args:
            distance_metric: Distance metric ('euclidean' or 'cosine')
            threshold: Anomaly threshold (higher = more sensitive)
            adaptive: Whether to use adaptive thresholding
            window_size: Window size for adaptive threshold
        """
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.adaptive = adaptive
        self.window_size = window_size
        
        # Reference features (running average)
        self.reference_features = None
        self.feature_history = deque(maxlen=window_size)
        
        # Statistics for adaptive threshold
        self.score_history = deque(maxlen=window_size)
        
        logger.info(f"Anomaly detector initialized: metric={distance_metric}, threshold={threshold}")
    
    def compute_distance(self, features: np.ndarray, reference: np.ndarray) -> float:
        """
        Compute distance between feature vectors.
        
        Args:
            features: Current feature vector
            reference: Reference feature vector
        
        Returns:
            Distance score
        """
        if self.distance_metric == "euclidean":
            distance = np.linalg.norm(features - reference)
        elif self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            dot_product = np.dot(features, reference)
            norm = np.linalg.norm(features) * np.linalg.norm(reference)
            cosine_similarity = dot_product / (norm + 1e-8)
            distance = 1.0 - cosine_similarity
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return float(distance)
    
    def update_reference(self, features: np.ndarray):
        """
        Update reference features using running average.
        
        Args:
            features: New feature vector
        """
        if self.reference_features is None:
            self.reference_features = features.copy()
        else:
            # Exponential moving average
            alpha = 0.1
            self.reference_features = (1 - alpha) * self.reference_features + alpha * features
        
        self.feature_history.append(features.copy())
    
    def detect(self, features: np.ndarray) -> tuple[float, bool]:
        """
        Detect if features are anomalous.
        
        Args:
            features: Feature vector to evaluate
        
        Returns:
            Tuple of (anomaly_score, is_anomalous)
        """
        if self.reference_features is None:
            # First frame - initialize and return normal
            self.update_reference(features)
            return 0.0, False
        
        # Compute distance from reference
        score = self.compute_distance(features, self.reference_features)
        
        # Update reference (only if not anomalous)
        if score < self.threshold:
            self.update_reference(features)
        
        # Adaptive threshold
        if self.adaptive:
            self.score_history.append(score)
            if len(self.score_history) > 10:
                mean_score = np.mean(self.score_history)
                std_score = np.std(self.score_history)
                adaptive_threshold = mean_score + 2 * std_score
                is_anomalous = score > adaptive_threshold
            else:
                is_anomalous = score > self.threshold
        else:
            is_anomalous = score > self.threshold
        
        return score, is_anomalous
    
    def reset(self):
        """Reset detector state."""
        self.reference_features = None
        self.feature_history.clear()
        self.score_history.clear()
        logger.info("Anomaly detector reset")


class AnomalyScorer:
    """
    Anomaly scorer that provides calibrated scores.
    
    Normalizes distance scores to [0, 1] range for better interpretation.
    """
    
    def __init__(self, detector: DistanceAnomalyDetector):
        """
        Initialize scorer.
        
        Args:
            detector: Base anomaly detector
        """
        self.detector = detector
        self.max_distance = 10.0  # Expected maximum distance
        self.min_distance = 0.0
    
    def score(self, features: np.ndarray) -> tuple[float, float, bool]:
        """
        Get normalized anomaly score.
        
        Args:
            features: Feature vector
        
        Returns:
            Tuple of (raw_score, normalized_score, is_anomalous)
        """
        raw_score, is_anomalous = self.detector.detect(features)
        
        # Normalize to [0, 1]
        normalized = (raw_score - self.min_distance) / (self.max_distance - self.min_distance + 1e-8)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return raw_score, normalized, is_anomalous
    
    def update_max_distance(self, new_max: float):
        """Update expected maximum distance."""
        self.max_distance = new_max
