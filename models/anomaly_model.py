"""
Top-level VigiLens anomaly detection model.

Integrates:
- Feature extraction (RGB/IR ResNet50)
- Modality-aware fusion
- Causal streaming LSTM
- Next-feature prediction

Trains on normal pedestrian behavior only.
Detects anomalies via prediction error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import logging

from models.feature_extractor import RGBFeatureExtractor, IRFeatureExtractor
from models.fusion import ModalityAwareFusion
from models.temporal_lstm import TemporalModel

logger = logging.getLogger(__name__)


class VigiLensModel(nn.Module):
    """
    Complete VigiLens anomaly detection model.

    Pipeline:
    RGB frame -> YOLO segmentation -> ROI -> ResNet50 -> 2048-D
    IR frame -> RGB-derived ROI -> ResNet50 -> 2048-D
    Fusion -> 512-D
    LSTM -> hidden state -> prediction head -> predicted next 512-D

    Training: Normal-only next-feature prediction
    Inference: Prediction error = anomaly score
    """

    def __init__(self,
                 device: str = "cuda",
                 input_size: Tuple[int, int] = (224, 224),
                 fusion_hidden_dim: int = 256,
                 fusion_output_dim: int = 512,
                 lstm_hidden_size: int = 256,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.2):
        """
        Initialize VigiLens model.

        Args:
            device: Device to run model on
            input_size: Input image size (H, W)
            fusion_hidden_dim: Hidden dimension for fusion projections
            fusion_output_dim: Output dimension after fusion (LSTM input)
            lstm_hidden_size: LSTM hidden size
            lstm_layers: Number of LSTM layers
            lstm_dropout: Dropout between LSTM layers
        """
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.fusion_output_dim = fusion_output_dim

        # Feature extractors
        self.rgb_extractor = RGBFeatureExtractor(device, input_size)
        self.ir_extractor = IRFeatureExtractor(device, input_size)

        # Modality-aware fusion
        self.fusion = ModalityAwareFusion(
            input_dim=2048,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim
        )

        # Temporal model (LSTM + prediction head)
        self.temporal = TemporalModel(
            input_size=fusion_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout
        )

        # Current hidden state for streaming inference
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Previous prediction for scoring
        self.previous_prediction: Optional[torch.Tensor] = None

        logger.info("VigiLens model initialized")

    def extract_features(self,
                        rgb_frame,
                        ir_frame=None,
                        rgb_mask=None,
                        ir_mask=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features from RGB and IR frames.

        Args:
            rgb_frame: RGB frame (H, W, 3)
            ir_frame: IR frame (H, W) or (H, W, 1) or None
            rgb_mask: Binary mask for RGB ROI (H, W) or None
            ir_mask: Binary mask for IR ROI (H, W) or None

        Returns:
            Tuple of (rgb_features, ir_features) as tensors
        """
        # Extract RGB features
        try:
            rgb_features_np = self.rgb_extractor.extract(rgb_frame, rgb_mask)
            rgb_features = torch.from_numpy(rgb_features_np).float().unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"RGB feature extraction failed: {e}")
            rgb_features = torch.zeros(1, 2048).to(self.device)

        # Extract IR features if available
        ir_features = None
        if ir_frame is not None:
            try:
                ir_features_np = self.ir_extractor.extract(ir_frame, ir_mask)
                ir_features = torch.from_numpy(ir_features_np).float().unsqueeze(0).to(self.device)
            except Exception as e:
                logger.error(f"IR feature extraction failed: {e}")
                ir_features = torch.zeros(1, 2048).to(self.device)

        return rgb_features, ir_features

    def fuse_features(self,
                     rgb_features: torch.Tensor,
                     ir_features: Optional[torch.Tensor],
                     rgb_available: bool = True,
                     ir_available: bool = True) -> torch.Tensor:
        """
        Fuse RGB and IR features with modality awareness.

        Args:
            rgb_features: RGB features (B, 2048)
            ir_features: IR features (B, 2048) or None
            rgb_available: Whether RGB is available
            ir_available: Whether IR is available

        Returns:
            Fused features (B, 512)
        """
        with torch.no_grad():
            fused = self.fusion(rgb_features, ir_features, rgb_available, ir_available)
        return fused

    def forward(self,
                fused_features: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through temporal model.

        Args:
            fused_features: Fused features (B, 512)
            hidden_state: Previous LSTM hidden state or None

        Returns:
            Tuple of (prediction, (h, c))
        """
        prediction, (h, c) = self.temporal(fused_features, hidden_state)
        return prediction, (h, c)

    def reset_stream(self):
        """Reset streaming state (hidden state and previous prediction)."""
        self.hidden_state = None
        self.previous_prediction = None
        logger.debug("Stream state reset")

    def process_frame(self,
                     rgb_frame,
                     ir_frame=None,
                     rgb_mask=None,
                     ir_mask=None,
                     rgb_available: bool = True,
                     ir_available: bool = True) -> Dict:
        """
        Process a single frame for streaming inference.

        Args:
            rgb_frame: RGB frame (H, W, 3)
            ir_frame: IR frame or None
            rgb_mask: RGB ROI mask or None
            ir_mask: IR ROI mask or None
            rgb_available: Whether RGB modality is available
            ir_available: Whether IR modality is available

       Returns:
            Dictionary containing:
            - prediction: Predicted next feature (for next frame)
            - anomaly_score: Prediction error from previous prediction
            - hidden_state: Current LSTM hidden state
        """
        with torch.no_grad():
            # Extract features
            rgb_features, ir_features = self.extract_features(rgb_frame, ir_frame, rgb_mask, ir_mask)

            # Fuse features
            fused = self.fuse_features(rgb_features, ir_features, rgb_available, ir_available)

            # Calculate anomaly score if we have a previous prediction
            anomaly_score = None
            if self.previous_prediction is not None:
                # Prediction error = distance between previous prediction and actual current feature
                # previous_prediction = predicted z_t (generated at frame t-1)
                # fused = actual z_t (current frame)
                error = F.mse_loss(self.previous_prediction, fused, reduction='none')
                anomaly_score = error.mean().item()

            # Pass through LSTM to generate prediction for NEXT frame
            prediction, (h, c) = self.forward(fused, self.hidden_state)

            # Update hidden state (detach to prevent gradient accumulation)
            self.hidden_state = (h.detach(), c.detach())

            # Store the CURRENT prediction for the NEXT frame
            # This prediction is for z_(t+1), will be compared against actual z_(t+1) at next frame
            self.previous_prediction = prediction.detach()

        return {
            'prediction': prediction,
            'anomaly_score': anomaly_score,
            'hidden_state': (h, c)
        }

    def training_step(self,
                     fused_features: torch.Tensor,
                     next_fused_features: torch.Tensor,
                     hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Training step: predict next feature and compute loss.

        Args:
            fused_features: Current features (B, 512)
            next_fused_features: Next frame features (B, 512) - target
            hidden_state: Previous hidden state or None

        Returns:
            Tuple of (loss, (h, c))
        """
        # Predict next feature
        prediction, (h, c) = self.forward(fused_features, hidden_state)

        # Compute prediction loss (Smooth L1 for robustness)
        loss = F.smooth_l1_loss(prediction, next_fused_features)

        return loss, (h, c)

    def save_checkpoint(self, path: str, epoch: int, loss: float, config: Dict):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            loss: Current loss
            config: Configuration dictionary
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'loss': loss,
            'config': config
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Dictionary containing epoch, loss, config
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")
        return {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'config': checkpoint['config']
        }
