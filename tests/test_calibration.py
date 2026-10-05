"""
Tests for anomaly calibration module.
"""

import numpy as np
from pathlib import Path
import json
import tempfile
from training.calibration import AnomalyCalibrator
from models.anomaly_model import VigiLensModel


def test_calibrator_compute_threshold():
    """Test threshold computation from errors."""
    model = VigiLensModel(device="cpu")
    calibrator = AnomalyCalibrator(model, k_std=3.0, percentile=95.0)

    # Generate synthetic errors
    errors = np.random.randn(100) * 0.1 + 0.5  # Mean 0.5, std 0.1

    calibration_info = calibrator.compute_threshold(errors)

    assert 'mean_error' in calibration_info
    assert 'std_error' in calibration_info
    assert 'threshold_std' in calibration_info
    assert 'threshold_percentile' in calibration_info

    # Threshold should be above mean
    assert calibration_info['threshold_std'] > calibration_info['mean_error']


def test_calibrator_save_load():
    """Test saving and loading calibration info."""
    model = VigiLensModel(device="cpu")
    calibrator = AnomalyCalibrator(model, k_std=3.0)

    calibration_info = {
        'mean_error': 0.5,
        'std_error': 0.1,
        'threshold_std': 0.8,
        'threshold_percentile': 0.75,
        'k_std': 3.0,
        'percentile': 95.0,
        'num_samples': 100,
        'method': 'std'
    }

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    calibrator.save_calibration(calibration_info, temp_path)

    # Load back
    loaded_info = AnomalyCalibrator.load_calibration(temp_path)

    assert loaded_info == calibration_info

    # Cleanup
    Path(temp_path).unlink()


def test_calibrator_k_std():
    """Test different k_std values."""
    model = VigiLensModel(device="cpu")
    calibrator = AnomalyCalibrator(model, k_std=2.0)

    errors = np.random.randn(100) * 0.1 + 0.5
    calibration_info = calibrator.compute_threshold(errors)

    # With k_std=2, threshold should be mean + 2*std
    expected_threshold = calibration_info['mean_error'] + 2.0 * calibration_info['std_error']
    assert abs(calibration_info['threshold_std'] - expected_threshold) < 0.01


def test_calibrator_percentile():
    """Test percentile-based threshold."""
    model = VigiLensModel(device="cpu")
    calibrator = AnomalyCalibrator(model, k_std=3.0, percentile=90.0)

    errors = np.random.randn(100) * 0.1 + 0.5
    calibration_info = calibrator.compute_threshold(errors)

    # Percentile threshold should match numpy percentile
    expected_percentile = np.percentile(errors, 90.0)
    assert abs(calibration_info['threshold_percentile'] - expected_percentile) < 0.01
