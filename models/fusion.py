"""
Multimodal fusion module for combining visible and IR features.

This module implements various fusion strategies to combine features from
visible and IR encoders for anomaly detection.
"""

import numpy as np
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)


class MultimodalFusion:
    """
    Multimodal fusion for combining visible and IR features.
    
    Supports multiple fusion strategies:
    - concat: Concatenate feature vectors
    - weighted: Weighted sum of features
    - average: Simple average of features
    """
    
    def __init__(self,
                 method: Literal["concat", "weighted", "average"] = "concat",
                 visible_weight: float = 0.6,
                 ir_weight: float = 0.4):
        """
        Initialize fusion module.
        
        Args:
            method: Fusion method ('concat', 'weighted', or 'average')
            visible_weight: Weight for visible features (for weighted fusion)
            ir_weight: Weight for IR features (for weighted fusion)
        """
        self.method = method
        self.visible_weight = visible_weight
        self.ir_weight = ir_weight
        
        if method == "weighted":
            # Normalize weights
            total = visible_weight + ir_weight
            self.visible_weight /= total
            self.ir_weight /= total
        
        logger.info(f"Fusion method: {method}")
        if method == "weighted":
            logger.info(f"Visible weight: {self.visible_weight:.2f}, IR weight: {self.ir_weight:.2f}")
    
    def fuse(self,
             visible_features: np.ndarray,
             ir_features: Optional[np.ndarray] = None,
             normalize: bool = True) -> np.ndarray:
        """
        Fuse visible and IR features.
        
        Args:
            visible_features: Visible feature vector (2048,)
            ir_features: IR feature vector (2048,) or None if IR unavailable
            normalize: Whether to L2-normalize features before fusion
        
        Returns:
            Fused feature vector
        
        Mathematical formulation:
        
        For concatenation:
            F_fused = concat(F_visible, F_IR)
            Dimension: 2048 + 2048 = 4096
        
        For weighted fusion:
            F_fused = alpha * F_visible + beta * F_IR
            Dimension: 2048
        
        For average fusion:
            F_fused = (F_visible + F_IR) / 2
            Dimension: 2048
        
        Note: Features are L2-normalized before fusion to ensure numerical
        comparability between modalities, especially for weighted/average fusion.
        """
        if ir_features is None:
            logger.warning("IR features unavailable, using visible only")
            return visible_features
        
        # Normalize features if requested (important for weighted/average fusion)
        if normalize:
            visible_features = self._l2_normalize(visible_features)
            ir_features = self._l2_normalize(ir_features)
        
        if self.method == "concat":
            # Concatenate features
            fused = np.concatenate([visible_features, ir_features])
            logger.debug(f"Fused shape (concat): {fused.shape}")
        
        elif self.method == "weighted":
            # Weighted sum
            fused = (self.visible_weight * visible_features +
                    self.ir_weight * ir_features)
            logger.debug(f"Fused shape (weighted): {fused.shape}")
        
        elif self.method == "average":
            # Simple average
            fused = (visible_features + ir_features) / 2.0
            logger.debug(f"Fused shape (average): {fused.shape}")
        
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
        
        return fused
    
    def _l2_normalize(self, features: np.ndarray) -> np.ndarray:
        """
        L2-normalize feature vector.
        
        Args:
            features: Feature vector
        
        Returns:
            L2-normalized feature vector
        """
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            return features / norm
        return features
    
    def get_output_dim(self, visible_dim: int, ir_dim: int) -> int:
        """
        Get output dimension based on fusion method.
        
        Args:
            visible_dim: Visible feature dimension
            ir_dim: IR feature dimension
        
        Returns:
            Output dimension after fusion
        """
        if self.method == "concat":
            return visible_dim + ir_dim
        else:
            # weighted and average preserve dimension
            return visible_dim


class LateFusion(MultimodalFusion):
    """
    Late fusion at feature level.
    
    Combines features after individual encoding.
    """
    
    def __init__(self, method: Literal["concat", "weighted", "average"] = "concat"):
        super().__init__(method)
        logger.info("Late fusion initialized")


class EarlyFusion:
    """
    Early fusion at input level.
    
    Combines visible and IR frames before feature extraction.
    This is not currently used but provided for completeness.
    """
    
    def __init__(self):
        logger.info("Early fusion initialized")
    
    def fuse_frames(self,
                   visible_frame: np.ndarray,
                   ir_frame: np.ndarray) -> np.ndarray:
        """
        Fuse visible and IR frames at input level.
        
        Args:
            visible_frame: Visible frame (H, W, 3)
            ir_frame: IR frame (H, W, 3) after conversion
        
        Returns:
            Fused frame (H, W, 6) - concatenated channels
        """
        # Concatenate along channel dimension
        fused = np.concatenate([visible_frame, ir_frame], axis=2)
        logger.debug(f"Fused frame shape: {fused.shape}")
        return fused
