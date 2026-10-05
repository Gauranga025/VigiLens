"""
Preprocessing module for visible and IR/thermal frames.

This module handles preprocessing of visible RGB frames and IR/thermal frames
for feature extraction. It provides proper IR handling without faking IR data.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FramePreprocessor:
    """
    Preprocessor for visible and IR frames.
    
    Handles resizing, normalization, and channel conversion for both modalities.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 ir_as_rgb: bool = True):
        """
        Initialize frame preprocessor.
        
        Args:
            target_size: Target (height, width) for frames
            normalize: Whether to normalize pixel values
            ir_as_rgb: Whether to convert IR to 3-channel for RGB-compatible encoders
        """
        self.target_size = target_size
        self.normalize = normalize
        self.ir_as_rgb = ir_as_rgb
        
        # ImageNet normalization parameters (for pretrained models)
        self.mean_visible = np.array([0.485, 0.456, 0.406])
        self.std_visible = np.array([0.229, 0.224, 0.225])
        
        # Simple normalization for IR
        self.mean_ir = np.array([0.5])
        self.std_ir = np.array([0.5])
    
    def preprocess_visible(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess visible RGB frame.
        
        Args:
            frame: BGR frame from OpenCV (H, W, 3)
        
        Returns:
            Preprocessed frame (H, W, 3) normalized to [0, 1] or standardized
        """
        if frame is None:
            raise ValueError("Visible frame is None")
        
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
        
        # Convert to float and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Standardize using ImageNet statistics
        if self.normalize:
            frame = (frame - self.mean_visible) / self.std_visible
        
        return frame
    
    def preprocess_ir(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess IR/thermal frame.
        
        IMPORTANT: This expects REAL IR/thermal data, not grayscale RGB.
        If IR is unavailable, this will raise an error rather than faking it.
        
        Args:
            frame: IR frame (H, W) or (H, W, 1) single-channel thermal data
        
        Returns:
            Preprocessed IR frame. If ir_as_rgb=True, returns (H, W, 3).
            Otherwise returns (H, W, 1).
        """
        if frame is None:
            raise ValueError("IR frame is None - IR input unavailable")
        
        # Ensure single channel
        if len(frame.shape) == 3:
            if frame.shape[2] == 3:
                logger.warning("IR frame appears to be RGB - this is not real thermal data")
                # Convert to grayscale but log warning
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif frame.shape[2] == 1:
                frame = frame.squeeze(axis=2)
        
        # Resize
        frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
        
        # Convert to float and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Standardize
        if self.normalize:
            frame = (frame - self.mean_ir) / self.std_ir
        
        # Add channel dimension
        frame = np.expand_dims(frame, axis=-1)  # (H, W, 1)
        
        # Convert to 3-channel if needed for RGB encoder
        if self.ir_as_rgb:
            # Replicate single channel to 3 channels
            frame = np.repeat(frame, 3, axis=2)  # (H, W, 3)
            logger.debug("IR converted to 3-channel for RGB-compatible encoder")
        
        return frame
    
    def preprocess_pair(self, 
                       visible_frame: np.ndarray, 
                       ir_frame: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess a visible-IR frame pair.
        
        Args:
            visible_frame: Visible RGB frame
            ir_frame: IR frame (None if unavailable)
        
        Returns:
            Tuple of (preprocessed_visible, preprocessed_ir)
            preprocessed_ir is None if ir_frame is None
        """
        visible = self.preprocess_visible(visible_frame)
        
        ir = None
        if ir_frame is not None:
            try:
                ir = self.preprocess_ir(ir_frame)
            except ValueError as e:
                logger.warning(f"IR preprocessing failed: {e}")
                ir = None
        
        return visible, ir


def validate_ir_frame(frame: np.ndarray) -> bool:
    """
    Validate that a frame is likely real IR/thermal data.
    
    This is a heuristic check - real thermal data typically has specific characteristics.
    However, this cannot definitively distinguish IR from grayscale RGB.
    
    Args:
        frame: Input frame to validate
    
    Returns:
        True if frame appears to be valid IR data
    """
    if frame is None:
        return False
    
    # Check if single channel
    if len(frame.shape) == 3 and frame.shape[2] > 1:
        logger.warning("Frame has multiple channels - unlikely to be pure IR")
        return False
    
    # Check data range (thermal data often has different ranges)
    if frame.dtype == np.uint16:
        # 16-bit thermal data is a strong indicator
        return True
    
    return True  # Cannot validate definitively, assume valid


def align_frames(visible_frame: np.ndarray, 
                ir_frame: np.ndarray,
                target_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align visible and IR frames to the same dimensions.
    
    This is a simple resize-based alignment. For pixel-perfect registration,
    geometric calibration parameters would be required.
    
    Args:
        visible_frame: Visible frame
        ir_frame: IR frame
        target_size: Target (height, width)
    
    Returns:
        Tuple of aligned (visible_frame, ir_frame)
    """
    visible_aligned = cv2.resize(visible_frame, (target_size[1], target_size[0]))
    ir_aligned = cv2.resize(ir_frame, (target_size[1], target_size[0]))
    
    return visible_aligned, ir_aligned
