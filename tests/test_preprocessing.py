"""
Tests for preprocessing module.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import FramePreprocessor


def test_preprocess_visible():
    """Test visible frame preprocessing."""
    preprocessor = FramePreprocessor(target_size=(224, 224))
    
    # Create dummy RGB frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess
    result = preprocessor.preprocess_visible(frame)
    
    # Check shape
    assert result.shape == (224, 224, 3), f"Expected (224, 224, 3), got {result.shape}"
    
    # Check range (normalized)
    assert result.min() >= 0.0 and result.max() <= 1.0, "Values should be in [0, 1]"
    
    print("✓ test_preprocess_visible passed")


def test_preprocess_ir():
    """Test IR frame preprocessing."""
    preprocessor = FramePreprocessor(target_size=(224, 224), ir_as_rgb=True)
    
    # Create dummy IR frame (single channel)
    frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    # Preprocess
    result = preprocessor.preprocess_ir(frame)
    
    # Check shape (should be 3-channel due to ir_as_rgb)
    assert result.shape == (224, 224, 3), f"Expected (224, 224, 3), got {result.shape}"
    
    # Check range
    assert result.min() >= 0.0 and result.max() <= 1.0, "Values should be in [0, 1]"
    
    print("✓ test_preprocess_ir passed")


def test_preprocess_pair():
    """Test preprocessing of visible-IR pair."""
    preprocessor = FramePreprocessor(target_size=(224, 224))
    
    # Create dummy frames
    visible = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    ir = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    # Preprocess pair
    vis_prep, ir_prep = preprocessor.preprocess_pair(visible, ir)
    
    # Check shapes
    assert vis_prep.shape == (224, 224, 3), f"Visible shape incorrect: {vis_prep.shape}"
    assert ir_prep.shape == (224, 224, 3), f"IR shape incorrect: {ir_prep.shape}"
    
    print("✓ test_preprocess_pair passed")


def test_ir_unavailable():
    """Test preprocessing when IR is unavailable."""
    preprocessor = FramePreprocessor(target_size=(224, 224))
    
    # Create only visible frame
    visible = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess with None IR
    vis_prep, ir_prep = preprocessor.preprocess_pair(visible, None)
    
    # Check visible is processed
    assert vis_prep.shape == (224, 224, 3)
    
    # Check IR is None
    assert ir_prep is None
    
    print("✓ test_ir_unavailable passed")


if __name__ == "__main__":
    test_preprocess_visible()
    test_preprocess_ir()
    test_preprocess_pair()
    test_ir_unavailable()
    print("\n✅ All preprocessing tests passed")
