"""
Anomaly calibration module.

Calculates normal prediction-error statistics and sets anomaly threshold.
Saves calibration information with the model checkpoint.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
import logging
from tqdm import tqdm

from models.anomaly_model import VigiLensModel
from training.dataset import NormalVideoDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class AnomalyCalibrator:
    """
    Calibrates anomaly detection threshold using normal validation data.

    Collects prediction errors from normal videos and computes statistics.
    """

    def __init__(self,
                 model: VigiLensModel,
                 k_std: float = 3.0,
                 percentile: float = 95.0):
        """
        Initialize calibrator.

        Args:
            model: Trained VigiLens model
            k_std: Number of standard deviations for threshold (mean + k * std)
            percentile: Percentile for threshold (alternative method)
        """
        self.model = model
        self.k_std = k_std
        self.percentile = percentile

    def collect_errors(self,
                      val_videos: List[Path],
                      val_ir_videos: List[Path],
                      device: str,
                      batch_size: int = 4) -> List[float]:
        """
        Collect prediction errors from normal validation videos.

        Args:
            val_videos: List of validation RGB video paths
            val_ir_videos: List of validation IR video paths
            device: Device
            batch_size: Batch size

        Returns:
            List of prediction errors
        """
        self.model.eval()
        self.model.reset_stream()

        errors = []

        # Create validation dataset
        val_dataset = NormalVideoDataset(
            val_videos,
            val_ir_videos,
            sequence_length=16,
            rgb_dropout_prob=0.0,
            ir_dropout_prob=0.0
        )

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Collecting errors"):
                rgb_sequence = batch['rgb_sequence']
                batch_size = rgb_sequence.shape[0]
                sequence_length = rgb_sequence.shape[1]

                # Reset stream for each video
                self.model.reset_stream()

                for t in range(sequence_length):
                    rgb_frame = rgb_sequence[:, t]
                    ir_frame = batch['ir_sequence'][:, t] if batch['ir_sequence'] is not None else None

                    # Placeholder features (real implementation uses actual extractors)
                    fused = torch.randn(batch_size, 512).to(device)

                    # Process frame
                    result = self.model.process_frame(
                        rgb_frame.cpu().numpy(),
                        ir_frame.cpu().numpy() if ir_frame is not None else None,
                        rgb_available=True,
                        ir_available=ir_frame is not None
                    )

                    if result['anomaly_score'] is not None:
                        errors.append(result['anomaly_score'])

        return errors

    def compute_threshold(self, errors: List[float]) -> Dict:
        """
        Compute anomaly threshold from error statistics.

        Args:
            errors: List of prediction errors from normal data

        Returns:
            Dictionary containing calibration statistics
        """
        errors = np.array(errors)

        mean_error = np.mean(errors)
        std_error = np.std(errors)
        median_error = np.median(errors)

        # Threshold methods
        threshold_std = mean_error + self.k_std * std_error
        threshold_percentile = np.percentile(errors, self.percentile)

        calibration_info = {
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'median_error': float(median_error),
            'threshold_std': float(threshold_std),
            'threshold_percentile': float(threshold_percentile),
            'k_std': self.k_std,
            'percentile': self.percentile,
            'num_samples': len(errors),
            'method': 'std'  # Default to std method
        }

        logger.info(f"Calibration: mean={mean_error:.4f}, std={std_error:.4f}")
        logger.info(f"Threshold (std): {threshold_std:.4f}")
        logger.info(f"Threshold (percentile): {threshold_percentile:.4f}")

        return calibration_info

    def save_calibration(self, calibration_info: Dict, save_path: str):
        """
        Save calibration information to JSON file.

        Args:
            calibration_info: Calibration statistics dictionary
            save_path: Path to save calibration JSON
        """
        with open(save_path, 'w') as f:
            json.dump(calibration_info, f, indent=2)

        logger.info(f"Calibration saved to {save_path}")

    @staticmethod
    def load_calibration(load_path: str) -> Dict:
        """
        Load calibration information from JSON file.

        Args:
            load_path: Path to calibration JSON file

        Returns:
            Calibration statistics dictionary
        """
        with open(load_path, 'r') as f:
            calibration_info = json.load(f)

        logger.info(f"Calibration loaded from {load_path}")
        return calibration_info


def calibrate_model(model: VigiLensModel,
                   val_videos: List[Path],
                   val_ir_videos: List[Path],
                   device: str,
                   checkpoint_dir: str = "checkpoints",
                   k_std: float = 3.0) -> Dict:
    """
    Calibrate model and save calibration with checkpoint.

    Args:
        model: Trained VigiLens model
        val_videos: List of validation RGB video paths
        val_ir_videos: List of validation IR video paths
        device: Device
        checkpoint_dir: Directory containing model checkpoint
        k_std: Number of standard deviations for threshold

    Returns:
        Calibration information dictionary
    """
    calibrator = AnomalyCalibrator(model, k_std=k_std)

    # Collect errors
    errors = calibrator.collect_errors(val_videos, val_ir_videos, device)

    # Compute threshold
    calibration_info = calibrator.compute_threshold(errors)

    # Save calibration
    calibration_path = Path(checkpoint_dir) / "calibration.json"
    calibrator.save_calibration(calibration_info, str(calibration_path))

    return calibration_info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config.config import SystemConfig

    config = SystemConfig()
    device = config.model.device

    # Load model
    model = VigiLensModel(device=device).to(device)
    model.load_checkpoint("checkpoints/best_model.pth")

    # Calibrate
    val_videos = [Path("Data/val/rgb/video1.mp4")]
    val_ir_videos = [Path("Data/val/ir/video1.mp4")]

    calibration_info = calibrate_model(model, val_videos, val_ir_videos, device)
    logger.info(f"Calibration complete: {calibration_info}")
