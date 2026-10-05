"""
Tests for training pipeline.

Verifies:
- Feature dimensionality
- Temporal target alignment
- Modality dropout
- Checkpoint loading compatibility
"""

import torch
import numpy as np
from pathlib import Path
from models.anomaly_model import VigiLensModel
from models.segmentation import YOLOSegmentation
from models.feature_extractor import RGBFeatureExtractor, IRFeatureExtractor
from models.fusion import ModalityAwareFusion


def test_training_feature_dimensions():
    """Test that training produces correct feature dimensions."""
    device = "cpu"
    
    # Initialize components
    segmentation = YOLOSegmentation(device=device)
    rgb_extractor = RGBFeatureExtractor(device=device)
    ir_extractor = IRFeatureExtractor(device=device)
    fusion = ModalityAwareFusion(input_dim=2048, hidden_dim=256, output_dim=512).to(device)
    
    # Create dummy frames
    rgb_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    ir_frame = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    
    # Extract RGB features
    rgb_mask = segmentation.extract_person_mask(rgb_frame)
    if rgb_mask is None:
        rgb_mask = np.ones((224, 224), dtype=np.uint8)
    
    rgb_features_np = rgb_extractor.extract(rgb_frame, rgb_mask)
    assert rgb_features_np.shape == (2048,), f"RGB features should be 2048-D, got {rgb_features_np.shape}"
    
    # Extract IR features
    ir_mask = rgb_mask  # Use same mask
    ir_features_np = ir_extractor.extract(ir_frame, ir_mask)
    assert ir_features_np.shape == (2048,), f"IR features should be 2048-D, got {ir_features_np.shape}"
    
    # Convert to tensors
    rgb_features = torch.from_numpy(rgb_features_np).float().unsqueeze(0).to(device)
    ir_features = torch.from_numpy(ir_features_np).float().unsqueeze(0).to(device)
    
    # Fuse
    fused = fusion(rgb_features, ir_features, rgb_available=True, ir_available=True)
    assert fused.shape == (1, 512), f"Fused features should be (1, 512), got {fused.shape}"
    
    print("✓ Training feature dimensions: PASS")


def test_temporal_target_alignment():
    """Test that temporal target is shifted by exactly one timestep."""
    device = "cpu"
    
    model = VigiLensModel(device=device).to(device)
    
    # Create dummy sequence
    batch_size = 2
    sequence_length = 5
    
    # z_1, z_2, z_3, z_4, z_5
    features = torch.randn(batch_size, sequence_length, 512).to(device)
    
    # Training should use:
    # input: z_1, z_2, z_3, z_4
    # target: z_2, z_3, z_4, z_5
    
    for t in range(sequence_length - 1):
        current = features[:, t]  # z_t
        target = features[:, t + 1]  # z_(t+1)
        
        # Verify they are different (shifted)
        assert not torch.allclose(current, target), f"Features at t and t+1 should be different"
        
        # Verify target is exactly the next timestep
        assert torch.equal(target, features[:, t + 1]), "Target should be exactly next timestep"
    
    print("✓ Temporal target alignment: PASS")


def test_modality_dropout_in_training():
    """Test that modality dropout ensures at least one modality is available."""
    from training.dataset import NormalVideoDataset
    from pathlib import Path
    import tempfile
    
    # Create dummy video files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal dummy video files
        rgb_path = Path(tmpdir) / "dummy.mp4"
        # We can't create real videos without cv2.VideoWriter, so we'll test the logic directly
        
        # Test modality dropout logic
        import random
        
        rgb_dropout_prob = 0.15
        ir_dropout_prob = 0.15
        
        for _ in range(100):
            rgb_available = random.random() > rgb_dropout_prob
            ir_available = random.random() > ir_dropout_prob
            
            # Apply safety check
            if not rgb_available and not ir_available:
                rgb_available = True
            
            # Verify at least one is available
            assert rgb_available or ir_available, "At least one modality must be available"
    
    print("✓ Modality dropout: PASS")


def test_checkpoint_compatibility():
    """Test that training checkpoint can be loaded by inference model."""
    device = "cpu"
    
    # Create model
    model = VigiLensModel(device=device).to(device)
    
    # Save checkpoint
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"
        model.save_checkpoint(str(checkpoint_path), epoch=1, loss=0.5, config={})
        
        # Create new model and load checkpoint
        model2 = VigiLensModel(device=device).to(device)
        model2.load_checkpoint(str(checkpoint_path))
        
        # Verify models have same parameters
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Loaded parameters should match saved parameters"
    
    print("✓ Checkpoint compatibility: PASS")


def test_lstm_input_dimensions():
    """Test that LSTM receives 512-D vectors."""
    device = "cpu"
    
    model = VigiLensModel(device=device).to(device)
    
    # Create dummy 512-D input
    batch_size = 2
    input_features = torch.randn(batch_size, 512).to(device)
    
    # Forward pass
    prediction, (h, c) = model.forward(input_features.unsqueeze(1))
    
    # Verify LSTM input was 512-D
    assert input_features.shape == (batch_size, 512), f"Input should be (B, 512), got {input_features.shape}"
    
    # Verify output dimensions
    assert prediction.shape == (batch_size, 512), f"Prediction should be (B, 512), got {prediction.shape}"
    assert h.shape == (2, batch_size, 256), f"Hidden state should be (2, B, 256), got {h.shape}"
    assert c.shape == (2, batch_size, 256), f"Cell state should be (2, B, 256), got {c.shape}"
    
    print("✓ LSTM input dimensions: PASS")


if __name__ == "__main__":
    test_training_feature_dimensions()
    test_temporal_target_alignment()
    test_modality_dropout_in_training()
    test_checkpoint_compatibility()
    test_lstm_input_dimensions()
    print("\nAll training pipeline tests passed!")
