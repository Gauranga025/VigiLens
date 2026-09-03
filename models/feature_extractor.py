"""
Feature extraction module using pretrained CNN encoders.

This module provides pretrained feature extractors for visible and IR frames
using ResNet50 from torchvision. No training is required - models use
ImageNet-pretrained weights.

Model Choice Rationale:
- ResNet50 is a well-established, widely-used architecture
- Pretrained on ImageNet (1.2M images, 1000 classes)
- Provides robust feature extraction without project-specific training
- Good balance between accuracy and inference speed
- Easy to integrate with existing PyTorch ecosystem

Limitations:
- IR frames are converted to 3-channel to work with RGB encoder
- The encoder was trained on RGB images, not thermal data
- This is a pragmatic compromise given lack of IR-specific pretrained models
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PretrainedFeatureExtractor(nn.Module):
    """
    Pretrained feature extractor based on ResNet50.
    
    Uses torchvision ResNet50 with ImageNet pretrained weights.
    Removes the final classification layer to extract features.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize pretrained feature extractor.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
        """
        super().__init__()
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load pretrained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Set to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        # Feature dimension
        self.feature_dim = 2048
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        logger.info("Pretrained ResNet50 feature extractor loaded")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input tensor.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Feature tensor (B, 2048)
        """
        with torch.no_grad():
            features = self.model(x)
            features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims
        return features
    
    def extract_from_numpy(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from numpy array frame.
        
        Args:
            frame: Preprocessed frame (H, W, C) normalized to ImageNet stats
        
        Returns:
            Feature vector (2048,)
        """
        # Convert to tensor
        if len(frame.shape) == 3:
            # (H, W, C) -> (C, H, W)
            frame = np.transpose(frame, (2, 0, 1))
        
        tensor = torch.from_numpy(frame).float().unsqueeze(0)
        tensor = tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.forward(tensor)
        
        return features.cpu().numpy().squeeze()


class VisibleFeatureExtractor(PretrainedFeatureExtractor):
    """
    Feature extractor for visible RGB frames.
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        logger.info("Visible feature extractor initialized")


class IRFeatureExtractor(PretrainedFeatureExtractor):
    """
    Feature extractor for IR/thermal frames.
    
    IMPORTANT: This uses the same ResNet50 architecture as visible extractor.
    IR frames are converted to 3-channel (by replication) to work with the
    RGB encoder. This is a limitation - the encoder was not trained on thermal data.
    
    This is a pragmatic approach given:
    1. Lack of widely-available IR-specific pretrained models
    2. Need for a no-training-required solution
    3. ResNet50's ability to extract general visual features
    
    The system will still benefit from multimodal fusion because:
    - IR provides different information (thermal vs RGB)
    - Features are extracted from different modalities
    - Fusion combines complementary information
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        logger.info("IR feature extractor initialized")
        logger.warning(
            "IR extractor uses RGB-pretrained ResNet50. "
            "IR frames are converted to 3-channel. "
            "This is a limitation - encoder not trained on thermal data."
        )


class FeatureExtractorFactory:
    """Factory for creating feature extractors."""
    
    @staticmethod
    def create_visible_extractor(device: str = "cuda") -> VisibleFeatureExtractor:
        """Create visible feature extractor."""
        return VisibleFeatureExtractor(device)
    
    @staticmethod
    def create_ir_extractor(device: str = "cuda") -> IRFeatureExtractor:
        """Create IR feature extractor."""
        return IRFeatureExtractor(device)
    
    @staticmethod
    def create_dual_extractors(device: str = "cuda") -> Tuple[VisibleFeatureExtractor, IRFeatureExtractor]:
        """Create both visible and IR extractors."""
        visible = VisibleFeatureExtractor(device)
        ir = IRFeatureExtractor(device)
        return visible, ir


def extract_features_batch(extractor: PretrainedFeatureExtractor,
                          frames: np.ndarray) -> np.ndarray:
    """
    Extract features from a batch of frames.
    
    Args:
        extractor: Feature extractor instance
        frames: Batch of frames (B, H, W, C)
    
    Returns:
        Feature vectors (B, 2048)
    """
    # Convert to tensor
    frames = np.transpose(frames, (0, 3, 1, 2))  # (B, C, H, W)
    tensor = torch.from_numpy(frames).float()
    tensor = tensor.to(extractor.device)
    
    # Extract features
    with torch.no_grad():
        features = extractor(tensor)
    
    return features.cpu().numpy()
