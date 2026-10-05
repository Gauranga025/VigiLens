"""
Tests for modality-aware fusion module.
"""

import torch
import numpy as np
from models.fusion import ModalityAwareFusion


def test_fusion_output_shape():
    """Test that fusion output has correct shape."""
    fusion = ModalityAwareFusion(input_dim=2048, hidden_dim=256, output_dim=512)
    rgb_features = torch.randn(2, 2048)
    ir_features = torch.randn(2, 2048)

    fused = fusion(rgb_features, ir_features, rgb_available=True, ir_available=True)
    assert fused.shape == (2, 512)


def test_rgb_only_fusion():
    """Test fusion with RGB only."""
    fusion = ModalityAwareFusion(input_dim=2048, hidden_dim=256, output_dim=512)
    rgb_features = torch.randn(2, 2048)

    fused = fusion(rgb_features, None, rgb_available=True, ir_available=False)
    assert fused.shape == (2, 512)


def test_ir_only_fusion():
    """Test fusion with IR only."""
    fusion = ModalityAwareFusion(input_dim=2048, hidden_dim=256, output_dim=512)
    ir_features = torch.randn(2, 2048)

    fused = fusion(torch.randn(2, 2048), ir_features, rgb_available=False, ir_available=True)
    assert fused.shape == (2, 512)


def test_both_modalities_fusion():
    """Test fusion with both RGB and IR."""
    fusion = ModalityAwareFusion(input_dim=2048, hidden_dim=256, output_dim=512)
    rgb_features = torch.randn(2, 2048)
    ir_features = torch.randn(2, 2048)

    fused = fusion(rgb_features, ir_features, rgb_available=True, ir_available=True)
    assert fused.shape == (2, 512)


def test_no_modalities_fusion():
    """Test fusion with neither modality (should return zeros)."""
    fusion = ModalityAwareFusion(input_dim=2048, hidden_dim=256, output_dim=512)
    rgb_features = torch.randn(2, 2048)

    fused = fusion(rgb_features, None, rgb_available=False, ir_available=False)
    assert fused.shape == (2, 512)
    assert torch.allclose(fused, torch.zeros_like(fused))


def test_get_output_dim():
    """Test get_output_dim method."""
    fusion = ModalityAwareFusion(input_dim=2048, hidden_dim=256, output_dim=512)
    assert fusion.get_output_dim() == 512
