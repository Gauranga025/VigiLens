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
    Distance-based anomaly detector with explicit calibration/inference phases.
    
    Computes anomaly score as distance from reference features.
    Separates calibration phase (building normal reference) from inference phase
    (detecting anomalies against frozen reference).
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
        
        # Reference features (frozen during inference)
        self.reference_features = None
        self.reference_built = False
        
        # Calibration data
        self.calibration_features = []
        self.calibration_window_size = 100
        
        # Statistics for adaptive threshold
        self.score_history = deque(maxlen=window_size)
        
        # Mode: 'calibration' or 'inference'
        self.mode = 'calibration'
        
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
    
    def start_calibration(self):
        """Start calibration phase to build normal reference."""
        self.mode = 'calibration'
        self.calibration_features = []
        self.reference_built = False
        logger.info("Started calibration phase")
    
    def add_calibration_sample(self, features: np.ndarray):
        """
        Add a calibration sample (assumed normal behavior).
        
        Args:
            features: Feature vector from normal frame
        """
        if self.mode != 'calibration':
            logger.warning("Not in calibration mode, sample ignored")
            return
        
        self.calibration_features.append(features.copy())
        
        if len(self.calibration_features) >= self.calibration_window_size:
            self.build_reference()
    
    def build_reference(self):
        """Build reference from calibration samples and switch to inference."""
        if len(self.calibration_features) == 0:
            logger.warning("No calibration samples, using mean of zeros")
            self.reference_features = np.zeros(2048)
        else:
            # Use mean of calibration samples as reference
            self.reference_features = np.mean(self.calibration_features, axis=0)
        
        self.reference_built = True
        self.mode = 'inference'
        logger.info(f"Reference built from {len(self.calibration_features)} samples")
        logger.info("Switched to inference phase")
    
    def detect(self, features: np.ndarray) -> tuple[float, bool]:
        """
        Detect if features are anomalous.
        
        Args:
            features: Feature vector to evaluate
        
        Returns:
            Tuple of (anomaly_score, is_anomalous)
        """
        if self.mode == 'calibration':
            # During calibration, add sample and return normal
            self.add_calibration_sample(features)
            return 0.0, False
        
        if not self.reference_built:
            logger.warning("Reference not built, auto-building from current frame")
            self.build_reference()
            return 0.0, False
        
        # Compute distance from reference
        score = self.compute_distance(features, self.reference_features)
        
        # Reference is frozen during inference - no updates
        # This prevents anomalies from contaminating the reference
        
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
        """Reset detector state and return to calibration mode."""
        self.reference_features = None
        self.reference_built = False
        self.calibration_features = []
        self.feature_history.clear()
        self.score_history.clear()
        self.mode = 'calibration'
        logger.info("Anomaly detector reset - returned to calibration mode")


class AnomalyScorer:
    """
    Anomaly scorer with statistical calibration.
    
    Provides calibrated anomaly scores based on calibration data statistics.
    Separates raw distance from normalized score for transparency.
    """
    
    def __init__(self, detector: DistanceAnomalyDetector):
        """
        Initialize scorer.
        
        Args:
            detector: Base anomaly detector
        """
        self.detector = detector
        
        # Calibration statistics
        self.calibration_distances = []
        self.mean_distance = None
        self.std_distance = None
        self.calibrated = False
    
    def calibrate(self, distances: list[float]):
        """
        Calibrate scorer using normal reference distances.
        
        Args:
            distances: List of distances from normal calibration samples
        """
        if len(distances) == 0:
            logger.warning("No calibration distances provided")
            return
        
        self.calibration_distances = distances.copy()
        self.mean_distance = np.mean(distances)
        self.std_distance = np.std(distances)
        self.calibrated = True
        
        logger.info(f"Scorer calibrated: mean={self.mean_distance:.4f}, std={self.std_distance:.4f}")
    
    def score(self, features: np.ndarray) -> tuple[float, float, bool]:
        """
        Get anomaly score with calibration.
        
        Args:
            features: Feature vector
        
        Returns:
            Tuple of (raw_distance, normalized_score, is_anomalous)
        """
        raw_distance, is_anomalous = self.detector.detect(features)
        
        if not self.calibrated:
            # Return raw distance if not calibrated
            return raw_distance, raw_distance, is_anomalous
        
        # Normalize using z-score: (distance - mean) / std
        if self.std_distance > 1e-8:
            normalized = (raw_distance - self.mean_distance) / self.std_distance
        else:
            normalized = raw_distance - self.mean_distance
        
        return raw_distance, normalized, is_anomalous
    
    def get_threshold_from_percentile(self, percentile: float = 95.0) -> float:
        """
        Get threshold based on calibration percentile.
        
        Args:
            percentile: Percentile (e.g., 95 for 95th percentile)
        
        Returns:
            Threshold value
        """
        if not self.calibrated:
            logger.warning("Scorer not calibrated, returning default threshold")
            return self.detector.threshold
        
        return np.percentile(self.calibration_distances, percentile)
