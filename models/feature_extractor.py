"""
Feature extraction module using pretrained ResNet50 encoders.

Extracts 2048-dimensional features from RGB and IR frames.
Supports ROI-based extraction using segmentation masks.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet50 feature extractor with ImageNet pretrained weights.

    Removes the final classification layer to extract 2048-D features.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize ResNet50 feature extractor.

        Args:
            device: Device to run model on ('cuda' or 'cpu')
        """
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.backbone.eval()
        self.backbone.to(self.device)

        self.feature_dim = 2048

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        logger.info(f"ResNet50 feature extractor loaded on {self.device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input tensor.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Feature tensor (B, 2048)
        """
        with torch.no_grad():
            features = self.backbone(x)
            features = features.squeeze(-1).squeeze(-1)
        return features


class RGBFeatureExtractor:
    """
    RGB frame feature extractor with ROI support.

    Extracts features from RGB frames, optionally using a segmentation mask
    to focus on pedestrian regions.
    """

    def __init__(self, device: str = "cuda", input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize RGB feature extractor.

        Args:
            device: Device to run model on
            input_size: Target input size (H, W)
        """
        self.device = device
        self.input_size = input_size
        self.extractor = ResNet50FeatureExtractor(device)

        logger.info("RGB feature extractor initialized")

    def preprocess(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Preprocess RGB frame for feature extraction.

        Args:
            frame: RGB frame (H, W, 3)
            mask: Optional binary mask (H, W) for ROI

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask[:, :, np.newaxis]
            frame = frame * mask_expanded

        # Resize
        frame = cv2.resize(frame, self.input_size)

        # Convert to float and normalize
        frame = frame.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std

        # Convert to tensor (ensure float32)
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        frame = frame.to(self.device)

        return frame

    def extract(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract features from RGB frame.

        Args:
            frame: RGB frame (H, W, 3)
            mask: Optional binary mask (H, W) for ROI

        Returns:
            Feature vector (2048,)
        """
        preprocessed = self.preprocess(frame, mask)
        features = self.extractor(preprocessed)
        return features.cpu().numpy().squeeze()


class IRFeatureExtractor:
    """
    IR/thermal frame feature extractor with ROI support.

    Uses the same ResNet50 architecture as RGB extractor.
    IR frames are converted to 3-channel to work with RGB encoder.
    This is a limitation - the encoder was not trained on thermal data.
    """

    def __init__(self, device: str = "cuda", input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize IR feature extractor.

        Args:
            device: Device to run model on
            input_size: Target input size (H, W)
        """
        self.device = device
        self.input_size = input_size
        self.extractor = ResNet50FeatureExtractor(device)

        logger.info("IR feature extractor initialized")
        logger.warning(
            "IR extractor uses RGB-pretrained ResNet50. "
            "IR frames are converted to 3-channel. "
            "This is a limitation - encoder not trained on thermal data."
        )

    def preprocess(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Preprocess IR frame for feature extraction.

        Args:
            frame: IR frame (H, W) or (H, W, 1)
            mask: Optional binary mask (H, W) for ROI

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Ensure single channel
        if len(frame.shape) == 3:
            frame = frame[:, :, 0]

        # Apply mask if provided
        if mask is not None:
            frame = frame * mask

        # Resize
        frame = cv2.resize(frame, self.input_size)

        # Convert to float and normalize
        frame = frame.astype(np.float32) / 255.0

        # Replicate to 3 channels
        frame = np.stack([frame, frame, frame], axis=-1)

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std

        # Convert to tensor (ensure float32)
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        frame = frame.to(self.device)

        return frame

    def extract(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract features from IR frame.

        Args:
            frame: IR frame (H, W) or (H, W, 1)
            mask: Optional binary mask (H, W) for ROI

        Returns:
            Feature vector (2048,)
        """
        preprocessed = self.preprocess(frame, mask)
        features = self.extractor(preprocessed)
        return features.cpu().numpy().squeeze()

