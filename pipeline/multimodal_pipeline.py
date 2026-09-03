"""
Main multimodal pipeline for VigiLens anomaly detection.

This module orchestrates the entire pipeline:
- Frame reading from visible and IR sources
- Preprocessing
- Feature extraction
- Multimodal fusion
- Anomaly detection
- Temporal smoothing
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
import logging

from pipeline.frame_source import MultimodalFrameSource, create_frame_source
from utils.preprocessing import FramePreprocessor
from models.feature_extractor import VisibleFeatureExtractor, IRFeatureExtractor
from models.fusion import MultimodalFusion
from models.anomaly_detector import DistanceAnomalyDetector, AnomalyScorer
from pipeline.temporal import TemporalSmoother
from config.config import SystemConfig

logger = logging.getLogger(__name__)


class MultimodalAnomalyPipeline:
    """
    Complete multimodal anomaly detection pipeline.
    
    Pipeline stages:
    1. Frame reading (visible + IR)
    2. Preprocessing (resize, normalize)
    3. Feature extraction (pretrained ResNet50)
    4. Multimodal fusion (concat/weighted/average)
    5. Anomaly detection (distance-based)
    6. Temporal smoothing
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize pipeline.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Initialize components
        self.preprocessor = FramePreprocessor(
            target_size=(config.preprocessing.target_height,
                        config.preprocessing.target_width),
            normalize=config.preprocessing.normalize,
            ir_as_rgb=config.preprocessing.ir_as_rgb
        )
        
        # Initialize feature extractors
        device = config.model.device
        self.visible_extractor = VisibleFeatureExtractor(device)
        self.ir_extractor = IRFeatureExtractor(device)
        
        # Initialize fusion
        self.fusion = MultimodalFusion(
            method=config.fusion.fusion_method,
            visible_weight=config.fusion.visible_weight,
            ir_weight=config.fusion.ir_weight
        )
        
        # Initialize anomaly detector
        self.detector = DistanceAnomalyDetector(
            distance_metric=config.anomaly.distance_metric,
            threshold=config.anomaly.anomaly_threshold,
            adaptive=config.anomaly.adaptive_threshold,
            window_size=config.anomaly.threshold_window
        )
        
        # Initialize scorer
        self.scorer = AnomalyScorer(self.detector)
        
        # Initialize temporal smoother
        self.smoother = TemporalSmoother(
            method=config.temporal.smoothing_method,
            window_size=config.temporal.window_size,
            consecutive_frames=config.temporal.consecutive_frames,
            alpha=config.temporal.alpha
        )
        
        # Frame source (initialized later)
        self.frame_source: Optional[MultimodalFrameSource] = None
        
        # Statistics
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        logger.info("Multimodal pipeline initialized")
    
    def load_source(self, visible_source: str, ir_source: Optional[str] = None):
        """
        Load frame source.
        
        Args:
            visible_source: Path to visible video or directory
            ir_source: Path to IR video or directory (optional)
        """
        self.frame_source = create_frame_source(visible_source, ir_source)
        logger.info(f"Frame source loaded: {visible_source}")
        if ir_source:
            logger.info(f"IR source: {ir_source}")
        else:
            logger.warning("IR source not provided - visible only mode")
    
    def process_frame(self,
                     visible_frame: np.ndarray,
                     ir_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process a single frame pair.
        
        Args:
            visible_frame: Visible RGB frame
            ir_frame: IR frame (optional)
        
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        
        # Preprocess
        visible_prep, ir_prep = self.preprocessor.preprocess_pair(
            visible_frame, ir_frame
        )
        
        # Extract features
        visible_features = self.visible_extractor.extract_from_numpy(visible_prep)
        
        ir_features = None
        if ir_prep is not None:
            ir_features = self.ir_extractor.extract_from_numpy(ir_prep)
        
        # Fuse features
        fused_features = self.fusion.fuse(visible_features, ir_features)
        
        # Detect anomaly
        raw_score, norm_score, is_anomalous = self.scorer.score(fused_features)
        
        # Temporal smoothing
        smoothed_score, smoothed_anomalous = self.smoother.smooth(
            norm_score, is_anomalous
        )
        
        # Update statistics
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1
        
        return {
            'visible_features': visible_features,
            'ir_features': ir_features,
            'fused_features': fused_features,
            'raw_score': raw_score,
            'normalized_score': norm_score,
            'smoothed_score': smoothed_score,
            'is_anomalous': smoothed_anomalous,
            'inference_time': inference_time,
            'ir_available': ir_frame is not None
        }
    
    def process_video(self) -> Dict[str, Any]:
        """
        Process entire video from loaded source.
        
        Returns:
            Dictionary containing overall statistics
        """
        if self.frame_source is None:
            raise RuntimeError("Frame source not loaded. Call load_source() first.")
        
        results = []
        anomaly_count = 0
        
        while True:
            visible_frame, ir_frame = self.frame_source.read()
            
            if visible_frame is None:
                break
            
            result = self.process_frame(visible_frame, ir_frame)
            results.append(result)
            
            if result['is_anomalous']:
                anomaly_count += 1
        
        self.frame_source.release()
        
        # Compute statistics
        avg_inference_time = self.total_inference_time / self.frame_count if self.frame_count > 0 else 0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        anomaly_rate = anomaly_count / self.frame_count if self.frame_count > 0 else 0
        
        return {
            'total_frames': self.frame_count,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'results': results
        }
    
    def reset(self):
        """Reset pipeline state."""
        self.detector.reset()
        self.smoother.reset()
        self.frame_count = 0
        self.total_inference_time = 0.0
        logger.info("Pipeline reset")
    
    def get_fps(self) -> float:
        """Get current average FPS."""
        if self.total_inference_time > 0 and self.frame_count > 0:
            return self.frame_count / self.total_inference_time
        return 0.0
