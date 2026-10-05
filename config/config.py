"""
Configuration module for VigiLens multimodal anomaly detection system.

This module contains all configurable parameters for the system including:
- Model settings (ResNet50, LSTM, fusion)
- Training parameters
- Inference parameters
- Calibration parameters
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Feature extraction
    visible_encoder: str = "resnet50"
    ir_encoder: str = "resnet50"
    feature_dim: int = 2048
    input_size: tuple = (224, 224)
    
    # Fusion
    fusion_hidden_dim: int = 256
    fusion_output_dim: int = 512
    
    # LSTM
    lstm_hidden_size: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    
    # Device
    device: str = "cuda"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Dataset
    sequence_length: int = 16
    rgb_dropout_prob: float = 0.15
    ir_dropout_prob: float = 0.15
    
    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 100
    train_ratio: float = 0.8
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 10


@dataclass
class CalibrationConfig:
    """Configuration for anomaly calibration."""
    
    # Threshold method
    k_std: float = 3.0  # Number of standard deviations
    percentile: float = 95.0  # Percentile threshold
    
    # Calibration file
    calibration_path: str = "checkpoints/calibration.json"


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    
    # Temporal smoothing
    smoothing_window: int = 10
    smoothing_method: str = "moving_average"  # "moving_average" or "exponential"
    ema_alpha: float = 0.3
    
    # Segmentation
    yolo_model_path: str = "yolov8n-seg.pt"
    
    # Checkpoint
    checkpoint_path: str = "checkpoints/best_model.pth"


@dataclass
class SystemConfig:
    """Main configuration class combining all sub-configurations."""
    
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    inference: InferenceConfig = InferenceConfig()
    
    # System settings
    debug_mode: bool = False
    log_level: str = "INFO"


# Default configuration instance
default_config = SystemConfig()


def get_config() -> SystemConfig:
    """Get the default system configuration."""
    return default_config
