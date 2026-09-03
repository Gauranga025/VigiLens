"""
Tests for fusion module.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fusion import MultimodalFusion


def test_concat_fusion():
    """Test concatenation fusion."""
    fusion = MultimodalFusion(method="concat")
    
    # Create dummy features
    visible = np.random.randn(2048)
    ir = np.random.randn(2048)
    
    # Fuse
    result = fusion.fuse(visible, ir)
    
    # Check shape
    assert result.shape == (4096,), f"Expected (4096,), got {result.shape}"
    
    print("✓ test_concat_fusion passed")


def test_weighted_fusion():
    """Test weighted fusion."""
    fusion = MultimodalFusion(method="weighted", visible_weight=0.7, ir_weight=0.3)
    
    # Create dummy features
    visible = np.ones(2048)
    ir = np.zeros(2048)
    
    # Fuse
    result = fusion.fuse(visible, ir)
    
    # Check shape
    assert result.shape == (2048,), f"Expected (2048,), got {result.shape}"
    
    # Check values (should be close to 0.7)
    assert np.allclose(result, 0.7), f"Expected ~0.7, got {result[0]}"
    
    print("✓ test_weighted_fusion passed")


def test_average_fusion():
    """Test average fusion."""
    fusion = MultimodalFusion(method="average")
    
    # Create dummy features
    visible = np.ones(2048)
    ir = np.zeros(2048)
    
    # Fuse
    result = fusion.fuse(visible, ir)
    
    # Check shape
    assert result.shape == (2048,), f"Expected (2048,), got {result.shape}"
    
    # Check values (should be 0.5)
    assert np.allclose(result, 0.5), f"Expected 0.5, got {result[0]}"
    
    print("✓ test_average_fusion passed")


def test_ir_unavailable_fusion():
    """Test fusion when IR is unavailable."""
    fusion = MultimodalFusion(method="concat")
    
    # Create only visible features
    visible = np.random.randn(2048)
    
    # Fuse with None IR
    result = fusion.fuse(visible, None)
    
    # Should return visible only
    assert np.array_equal(result, visible), "Should return visible when IR unavailable"
    
    print("✓ test_ir_unavailable_fusion passed")


if __name__ == "__main__":
    test_concat_fusion()
    test_weighted_fusion()
    test_average_fusion()
    test_ir_unavailable_fusion()
    print("\n✅ All fusion tests passed")
