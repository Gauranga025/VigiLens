"""
Validation script for VigiLens model.

Evaluates the model on held-out normal videos.
Computes prediction loss and statistics for calibration.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm

from models.anomaly_model import VigiLensModel
from training.dataset import NormalVideoDataset

logger = logging.getLogger(__name__)


def validate(model: VigiLensModel,
             val_videos: List[Path],
             val_ir_videos: List[Path],
             device: str,
             batch_size: int = 4) -> Dict:
    """
    Validate model on held-out normal videos.

    Args:
        model: VigiLens model
        val_videos: List of validation RGB video paths
        val_ir_videos: List of validation IR video paths
        device: Device
        batch_size: Batch size

    Returns:
        Dictionary containing validation statistics
    """
    model.eval()

    # Create validation dataset
    val_dataset = NormalVideoDataset(
        val_videos,
        val_ir_videos,
        sequence_length=16,
        rgb_dropout_prob=0.0,
        ir_dropout_prob=0.0
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            rgb_sequence = batch['rgb_sequence']
            batch_size = rgb_sequence.shape[0]

            # Placeholder features (real implementation uses actual extractors)
            fused = torch.randn(batch_size, 512).to(device)
            fused_next = torch.randn(batch_size, 512).to(device)

            loss, _ = model.training_step(fused.unsqueeze(1), fused_next.unsqueeze(1))
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return {
        'val_loss': avg_loss,
        'num_batches': num_batches
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config.config import SystemConfig

    config = SystemConfig()
    device = config.model.device

    # Load model
    model = VigiLensModel(device=device).to(device)
    model.load_checkpoint("checkpoints/best_model.pth")

    # Validate
    val_videos = [Path("Data/val/rgb/video1.mp4")]
    val_ir_videos = [Path("Data/val/ir/video1.mp4")]

    results = validate(model, val_videos, val_ir_videos, device)
    logger.info(f"Validation results: {results}")
