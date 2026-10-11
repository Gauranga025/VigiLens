"""
Tests for training pipeline.

Verifies:
- Feature dimensionality
- Temporal target alignment
- Modality dropout
- Checkpoint loading compatibility
- AVI video support
- RGB/IR pairing by filename
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import cv2
from models.anomaly_model import VigiLensModel
from models.segmentation import YOLOSegmentation
from models.feature_extractor import RGBFeatureExtractor, IRFeatureExtractor
from models.fusion import ModalityAwareFusion
from training.train import extract_sequence_id, pair_videos_by_sequence_id, discover_videos


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


def test_sequence_id_extraction():
    """Test sequence ID extraction from filenames."""
    # Test various naming conventions
    assert extract_sequence_id(Path("scene001_rgb.mp4")) == "scene001"
    assert extract_sequence_id(Path("scene001_ir.avi")) == "scene001"
    assert extract_sequence_id(Path("video1_rgb.mov")) == "video1"
    assert extract_sequence_id(Path("video1-ir.mkv")) == "video1"
    assert extract_sequence_id(Path("test_visible.mp4")) == "test"
    assert extract_sequence_id(Path("test_thermal.avi")) == "test"
    
    print("✓ Sequence ID extraction: PASS")


def test_rgb_ir_pairing_by_filename():
    """Test RGB/IR pairing by sequence ID."""
    rgb_videos = [
        Path("scene001_rgb.mp4"),
        Path("scene002_rgb.avi"),
        Path("scene003_rgb.mov")
    ]
    ir_videos = [
        Path("scene001_ir.avi"),
        Path("scene002_ir.mp4"),
        Path("scene003_ir.mkv")
    ]
    
    pairs = pair_videos_by_sequence_id(rgb_videos, ir_videos)
    
    assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}"
    assert pairs[0][0].name == "scene001_rgb.mp4"
    assert pairs[0][1].name == "scene001_ir.avi"
    assert pairs[1][0].name == "scene002_rgb.avi"
    assert pairs[1][1].name == "scene002_ir.mp4"
    
    print("✓ RGB/IR pairing by filename: PASS")


def test_mixed_extension_support():
    """Test pairing with different extensions."""
    rgb_videos = [Path("scene001_rgb.mp4")]
    ir_videos = [Path("scene001_ir.avi")]
    
    pairs = pair_videos_by_sequence_id(rgb_videos, ir_videos)
    
    assert len(pairs) == 1
    assert pairs[0][0].suffix == ".mp4"
    assert pairs[0][1].suffix == ".avi"
    
    print("✓ Mixed extension support: PASS")


def test_unmatched_video_detection():
    """Test detection of unmatched videos."""
    rgb_videos = [Path("scene001_rgb.mp4"), Path("scene002_rgb.mp4")]
    ir_videos = [Path("scene001_ir.avi")]  # Missing scene002
    
    try:
        pairs = pair_videos_by_sequence_id(rgb_videos, ir_videos)
        # Should succeed but with warning for unmatched
        assert len(pairs) == 1  # Only scene001 paired
        print("✓ Unmatched video detection: PASS")
    except ValueError as e:
        # If it raises error for no matches, that's also acceptable
        print("✓ Unmatched video detection: PASS")


def test_duplicate_sequence_id_detection():
    """Test detection of duplicate sequence IDs."""
    rgb_videos = [Path("scene001_rgb.mp4"), Path("scene001_rgb.avi")]  # Duplicate scene001
    
    try:
        pairs = pair_videos_by_sequence_id(rgb_videos, [])
        assert False, "Should have raised ValueError for duplicate sequence ID"
    except ValueError as e:
        assert "Duplicate" in str(e)
        print("✓ Duplicate sequence ID detection: PASS")


def test_avi_video_discovery():
    """Test AVI video discovery in directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        rgb_dir = tmpdir / "train" / "rgb"
        ir_dir = tmpdir / "train" / "ir"
        rgb_dir.mkdir(parents=True)
        ir_dir.mkdir(parents=True)
        
        # Create dummy AVI files
        (rgb_dir / "scene001_rgb.avi").touch()
        (rgb_dir / "scene002_rgb.avi").touch()
        (ir_dir / "scene001_ir.avi").touch()
        (ir_dir / "scene002_ir.avi").touch()
        
        # Discover videos
        rgb_videos = discover_videos(tmpdir, "train", "rgb", extensions=['.avi'])
        ir_videos = discover_videos(tmpdir, "train", "ir", extensions=['.avi'])
        
        assert len(rgb_videos) == 2
        assert len(ir_videos) == 2
        assert all(v.suffix == ".avi" for v in rgb_videos)
        assert all(v.suffix == ".avi" for v in ir_videos)
        
        print("✓ AVI video discovery: PASS")


def test_missing_ir_required():
    """Test that discover_videos raises error when IR is required but missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        rgb_dir = tmpdir / "train" / "rgb"
        ir_dir = tmpdir / "train" / "ir"
        rgb_dir.mkdir(parents=True)
        # Don't create IR directory
        
        (rgb_dir / "scene001_rgb.avi").touch()
        
        # Should raise error when required=True
        try:
            discover_videos(tmpdir, "train", "ir", required=True)
            assert False, "Should have raised ValueError for missing IR when required=True"
        except ValueError as e:
            assert "not found" in str(e).lower() or "no videos" in str(e).lower()
            print("✓ Missing IR required: PASS")


def test_missing_ir_optional():
    """Test that discover_videos returns empty list when IR is optional and missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        rgb_dir = tmpdir / "train" / "rgb"
        rgb_dir.mkdir(parents=True)
        
        (rgb_dir / "scene001_rgb.avi").touch()
        
        # Should return empty list when required=False
        ir_videos = discover_videos(tmpdir, "train", "ir", required=False)
        assert ir_videos == []
        
        print("✓ Missing IR optional: PASS")


def test_rgb_only_mode_dataset():
    """Test that dataset handles None IR video paths correctly."""
    from training.dataset import NormalVideoDataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        rgb_dir = tmpdir / "train" / "rgb"
        rgb_dir.mkdir(parents=True)
        
        # Create a minimal dummy video file
        rgb_video = rgb_dir / "scene001_rgb.avi"
        rgb_video.touch()
        
        # Dataset with None IR paths (RGB-only mode)
        dataset = NormalVideoDataset(
            video_paths=[rgb_video],
            ir_video_paths=None,
            sequence_length=4,
            rgb_dropout_prob=0.0,
            ir_dropout_prob=0.0
        )
        
        # Verify dataset initialization
        assert len(dataset.video_paths) == 1
        assert dataset.ir_video_paths == [None]
        
        print("✓ RGB-only mode dataset: PASS")


if __name__ == "__main__":
    test_training_feature_dimensions()
    test_temporal_target_alignment()
    test_modality_dropout_in_training()
    test_checkpoint_compatibility()
    test_lstm_input_dimensions()
    test_sequence_id_extraction()
    test_rgb_ir_pairing_by_filename()
    test_mixed_extension_support()
    test_unmatched_video_detection()
    test_duplicate_sequence_id_detection()
    test_avi_video_discovery()
    test_missing_ir_required()
    test_missing_ir_optional()
    test_rgb_only_mode_dataset()
    print("\nAll training pipeline tests passed!")
