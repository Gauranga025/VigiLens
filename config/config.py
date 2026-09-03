"""
Configuration module for VigiLens multimodal anomaly detection system.

This module contains all configurable parameters for the system including:
- Model settings
- Preprocessing parameters
- Anomaly detection thresholds
- Temporal smoothing settings
- Fusion method configuration
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for pretrained feature extraction models."""
    
    # Visible encoder settings
    visible_encoder: str = "resnet50"  # ResNet50 pretrained on ImageNet
    visible_feature_dim: int = 2048  # ResNet50 output dimension
    
    # IR encoder settings
    ir_encoder: str = "resnet50"  # Using same architecture for IR
    ir_feature_dim: int = 2048
    
    # Model loading
    device: str = "cuda"  # Will fallback to CPU if CUDA unavailable
    pretrained_weights: bool = True


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing visible and IR frames."""
    
    # Target dimensions
    target_height: int = 224
    target_width: int = 224
    
    # Normalization
    normalize: bool = True
    mean_visible: tuple = (0.485, 0.456, 0.406)  # ImageNet mean
    std_visible: tuple = (0.229, 0.224, 0.225)   # ImageNet std
    mean_ir: tuple = (0.5,)  # For single-channel IR
    std_ir: tuple = (0.5,)
    
    # IR channel handling
    ir_as_rgb: bool = True  # Convert IR to 3-channel for RGB encoder
    ir_replication: bool = True  # Replicate IR channel to 3 channels


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion."""
    
    fusion_method: Literal["concat", "weighted", "average"] = "concat"
    
    # Weighted fusion parameters
    visible_weight: float = 0.6
    ir_weight: float = 0.4
    
    # Fusion output dimension
    fused_dim: int = 4096  # 2048 + 2048 for concatenation


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    
    # Anomaly scoring method
    scoring_method: Literal["distance", "reconstruction"] = "distance"
    
    # Distance-based scoring parameters
    reference_features_path: str = None  # Path to saved reference features
    distance_metric: Literal["euclidean", "cosine"] = "euclidean"
    
    # Thresholds
    anomaly_threshold: float = 0.65
    min_confidence: float = 0.5
    
    # Adaptive threshold
    adaptive_threshold: bool = False
    threshold_window: int = 100


@dataclass  
class TemporalConfig:
    """Configuration for temporal smoothing."""
    
    # Smoothing method
    smoothing_method: Literal["moving_average", "exponential", "consecutive"] = "moving_average"
    
    # Window parameters
    window_size: int = 10
    consecutive_frames: int = 3
    
    # Exponential smoothing
    alpha: float = 0.3  # Smoothing factor for exponential smoothing


@dataclass
class UIConfig:
    """Configuration for Streamlit UI."""
    
    # Display settings
    display_fps: bool = True
    display_frame_number: bool = True
    display_object_count: bool = True
    
    # Visualization
    show_bounding_boxes: bool = True
    show_anomaly_score: bool = True
    show_ir_frame: bool = True
    
    # Alert settings
    alert_display_frames: int = 5


@dataclass
class SystemConfig:
    """Main configuration class combining all sub-configurations."""
    
    model: ModelConfig = ModelConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    fusion: FusionConfig = FusionConfig()
    anomaly: AnomalyConfig = AnomalyConfig()
    temporal: TemporalConfig = TemporalConfig()
    ui: UIConfig = UIConfig()
    
    # System settings
    debug_mode: bool = False
    log_level: str = "INFO"


# Default configuration instance
default_config = SystemConfig()


def get_config() -> SystemConfig:
    """Get the default system configuration."""
    return default_config
