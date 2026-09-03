"""
Tests for anomaly detector calibration mechanism.
"""

import numpy as np
from models.anomaly_detector import DistanceAnomalyDetector, AnomalyScorer


def test_calibration_mode():
    """Test that calibration mode works correctly."""
    detector = DistanceAnomalyDetector(threshold=0.5)

    # Start calibration
    detector.start_calibration()
    assert detector.mode == 'calibration'
    assert not detector.reference_built

    # Add calibration samples
    for i in range(100):
        features = np.random.randn(2048)
        detector.add_calibration_sample(features)

    # Reference should be built after enough samples
    assert detector.reference_built
    assert detector.mode == 'inference'


def test_reference_frozen_during_inference():
    """Test that reference is frozen during inference."""
    detector = DistanceAnomalyDetector(threshold=0.5)

    # Calibrate
    detector.start_calibration()
    for i in range(100):
        features = np.random.randn(2048)
        detector.add_calibration_sample(features)

    # Save initial reference
    initial_reference = detector.reference_features.copy()

    # Run inference with anomalous features
    anomalous_features = np.random.randn(2048) * 10
    detector.detect(anomalous_features)

    # Reference should not change
    assert np.allclose(detector.reference_features, initial_reference)


def test_reset_returns_to_calibration():
    """Test that reset returns to calibration mode."""
    detector = DistanceAnomalyDetector(threshold=0.5)

    # Calibrate
    detector.start_calibration()
    for i in range(100):
        features = np.random.randn(2048)
        detector.add_calibration_sample(features)

    # Should be in inference mode
    assert detector.mode == 'inference'

    # Reset
    detector.reset()

    # Should return to calibration mode
    assert detector.mode == 'calibration'
    assert not detector.reference_built


def test_scorer_calibration():
    """Test statistical calibration of scorer."""
    detector = DistanceAnomalyDetector(threshold=0.5)
    scorer = AnomalyScorer(detector)

    # Create some calibration distances
    calibration_distances = [0.1, 0.15, 0.12, 0.14, 0.13] * 20

    # Calibrate scorer
    scorer.calibrate(calibration_distances)

    assert scorer.calibrated
    assert scorer.mean_distance is not None
    assert scorer.std_distance is not None


def test_raw_distance_preserved():
    """Test that raw distance is preserved."""
    detector = DistanceAnomalyDetector(threshold=0.5)
    scorer = AnomalyScorer(detector)

    # Calibrate
    detector.start_calibration()
    for i in range(100):
        features = np.random.randn(2048)
        detector.add_calibration_sample(features)

    calibration_distances = [0.1, 0.15, 0.12, 0.14, 0.13] * 20
    scorer.calibrate(calibration_distances)

    # Test frame
    test_features = np.random.randn(2048)
    raw_distance, norm_score, is_anomalous = scorer.score(test_features)

    # Raw distance should be a valid float
    assert isinstance(raw_distance, float)
    assert raw_distance >= 0


def test_threshold_from_percentile():
    """Test percentile-based threshold."""
    detector = DistanceAnomalyDetector(threshold=0.5)
    scorer = AnomalyScorer(detector)

    calibration_distances = [0.1, 0.15, 0.12, 0.14, 0.13] * 20
    scorer.calibrate(calibration_distances)

    # Get 95th percentile threshold
    threshold = scorer.get_threshold_from_percentile(95.0)

    assert isinstance(threshold, float)
    assert threshold >= min(calibration_distances)


if __name__ == "__main__":
    test_calibration_mode()
    test_reference_frozen_during_inference()
    test_reset_returns_to_calibration()
    test_scorer_calibration()
    test_raw_distance_preserved()
    test_threshold_from_percentile()
    print("All anomaly calibration tests passed!")
