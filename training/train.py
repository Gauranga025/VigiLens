"""
Training script for VigiLens normal-only anomaly detection.

Trains the model to predict next features from normal pedestrian behavior.
Uses Smooth L1 loss for robust prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import json
import numpy as np
import cv2
from tqdm import tqdm

from models.anomaly_model import VigiLensModel
from models.segmentation import YOLOSegmentation
from models.feature_extractor import RGBFeatureExtractor, IRFeatureExtractor
from models.fusion import ModalityAwareFusion
from training.dataset import NormalVideoDataset, split_videos_by_video
from config.config import SystemConfig

logger = logging.getLogger(__name__)


def train_epoch(model: VigiLensModel,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                device: str,
                epoch: int,
                segmentation: YOLOSegmentation,
                rgb_extractor: RGBFeatureExtractor,
                ir_extractor: IRFeatureExtractor,
                fusion: ModalityAwareFusion) -> float:
    """
    Train for one epoch.

    Args:
        model: VigiLens model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        segmentation: YOLO segmentation model
        rgb_extractor: RGB feature extractor
        ir_extractor: IR feature extractor
        fusion: Modality-aware fusion module

    Returns:
        Average loss for the epoch
    """
    model.train()
    # Set frozen components to eval mode
    segmentation.model.eval()
    rgb_extractor.extractor.eval()
    ir_extractor.extractor.eval()
    fusion.eval()  # Keep fusion frozen initially

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        rgb_sequence = batch['rgb_sequence']  # (B, T, H, W, 3)
        ir_sequence = batch['ir_sequence']  # (B, T, H, W) or None
        rgb_available = batch['rgb_available']  # (B,)
        ir_available = batch['ir_available']  # (B,)

        batch_size = rgb_sequence.shape[0]
        sequence_length = rgb_sequence.shape[1]

        # Initialize hidden state
        hidden_state = model.init_hidden(batch_size, device)

        # Process sequence
        batch_loss = 0.0
        for t in range(sequence_length - 1):
            # Current frame
            rgb_frame = rgb_sequence[:, t]  # (B, H, W, 3)
            ir_frame = ir_sequence[:, t] if ir_sequence is not None else None

            # Next frame (target)
            rgb_next = rgb_sequence[:, t + 1]
            ir_next = ir_sequence[:, t + 1] if ir_sequence is not None else None

            # Get modality availability for this timestep
            rgb_avail_t = rgb_available[t].item() if isinstance(rgb_available, torch.Tensor) else True
            ir_avail_t = ir_available[t].item() if isinstance(ir_available, torch.Tensor) and ir_frame is not None else False

            # Extract features for current frame with no_grad for frozen components
            with torch.no_grad():
                # Extract RGB features for current frame
                rgb_features_list = []
                rgb_masks_list = []
                for b in range(batch_size):
                    if rgb_avail_t:
                        rgb_np = rgb_frame[b].cpu().numpy().astype(np.uint8)
                        rgb_mask = segmentation.extract_person_mask(rgb_np)
                        rgb_features_np = rgb_extractor.extract(rgb_np, rgb_mask)
                        rgb_features_list.append(rgb_features_np)
                        rgb_masks_list.append(rgb_mask)
                    else:
                        rgb_features_list.append(np.zeros(2048))
                        rgb_masks_list.append(None)

                # Extract IR features for current frame
                ir_features_list = []
                ir_masks_list = []
                if ir_frame is not None:
                    for b in range(batch_size):
                        if ir_avail_t and rgb_masks_list[b] is not None:
                            ir_np = ir_frame[b].cpu().numpy().astype(np.uint8)
                            # Resize RGB mask to IR dimensions
                            ir_mask = cv2.resize(rgb_masks_list[b], (ir_np.shape[1], ir_np.shape[0]))
                            ir_features_np = ir_extractor.extract(ir_np, ir_mask)
                            ir_features_list.append(ir_features_np)
                            ir_masks_list.append(ir_mask)
                        elif ir_avail_t:
                            # IR-only mode: full frame
                            ir_np = ir_frame[b].cpu().numpy().astype(np.uint8)
                            ir_features_np = ir_extractor.extract(ir_np, None)
                            ir_features_list.append(ir_features_np)
                            ir_masks_list.append(None)
                        else:
                            ir_features_list.append(np.zeros(2048))
                            ir_masks_list.append(None)
                else:
                    ir_features_list = [np.zeros(2048)] * batch_size
                    ir_masks_list = [None] * batch_size

            # Convert to tensors
            rgb_features = torch.from_numpy(np.array(rgb_features_list)).float().to(device)
            ir_features = torch.from_numpy(np.array(ir_features_list)).float().to(device) if any(f.any() for f in ir_features_list) else None

            # Fuse features (trainable)
            fused = fusion(rgb_features, ir_features, rgb_avail_t, ir_avail_t)

            # Extract features for next frame (target) with no_grad
            with torch.no_grad():
                rgb_next_features_list = []
                rgb_next_masks_list = []
                for b in range(batch_size):
                    if rgb_avail_t:
                        rgb_next_np = rgb_next[b].cpu().numpy().astype(np.uint8)
                        rgb_next_mask = segmentation.extract_person_mask(rgb_next_np)
                        rgb_next_features_np = rgb_extractor.extract(rgb_next_np, rgb_next_mask)
                        rgb_next_features_list.append(rgb_next_features_np)
                    else:
                        rgb_next_features_list.append(np.zeros(2048))

                ir_next_features_list = []
                if ir_next is not None:
                    for b in range(batch_size):
                        if ir_avail_t and rgb_next_masks_list[b] is not None:
                            ir_next_np = ir_next[b].cpu().numpy().astype(np.uint8)
                            ir_next_mask = cv2.resize(rgb_next_masks_list[b], (ir_next_np.shape[1], ir_next_np.shape[0]))
                            ir_next_features_np = ir_extractor.extract(ir_next_np, ir_next_mask)
                            ir_next_features_list.append(ir_next_features_np)
                        elif ir_avail_t:
                            ir_next_np = ir_next[b].cpu().numpy().astype(np.uint8)
                            ir_next_features_np = ir_extractor.extract(ir_next_np, None)
                            ir_next_features_list.append(ir_next_features_np)
                        else:
                            ir_next_features_list.append(np.zeros(2048))
                else:
                    ir_next_features_list = [np.zeros(2048)] * batch_size

            # Convert to tensors
            rgb_next_features = torch.from_numpy(np.array(rgb_next_features_list)).float().to(device)
            ir_next_features = torch.from_numpy(np.array(ir_next_features_list)).float().to(device) if any(f.any() for f in ir_next_features_list) else None

            # Fuse next frame features
            fused_next = fusion(rgb_next_features, ir_next_features, rgb_avail_t, ir_avail_t)

            # Training step: z_t -> predict z_(t+1)
            loss, hidden_state = model.training_step(
                fused.unsqueeze(1),  # (B, 1, 512)
                fused_next.unsqueeze(1),  # (B, 1, 512)
                hidden_state
            )

            batch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss
        num_batches += 1

        pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train(config: SystemConfig,
          train_videos: list,
          val_videos: list,
          train_ir_videos: Optional[list] = None,
          val_ir_videos: Optional[list] = None,
          num_epochs: int = 100,
          batch_size: int = 4,
          learning_rate: float = 1e-4,
          checkpoint_dir: str = "checkpoints"):
    """
    Main training loop.

    Args:
        config: System configuration
        train_videos: List of training RGB video paths
        val_videos: List of validation RGB video paths
        train_ir_videos: List of training IR video paths
        val_ir_videos: List of validation IR video paths
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_dir: Directory to save checkpoints
    """
    device = config.model.device

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    # Initialize model
    model = VigiLensModel(device=device).to(device)

    # Initialize frozen components
    segmentation = YOLOSegmentation(device=device)
    rgb_extractor = RGBFeatureExtractor(device=device)
    ir_extractor = IRFeatureExtractor(device=device)
    fusion = ModalityAwareFusion(input_dim=2048, hidden_dim=256, output_dim=512).to(device)

    # Freeze backbones and fusion
    for param in segmentation.model.parameters():
        param.requires_grad = False
    for param in rgb_extractor.extractor.parameters():
        param.requires_grad = False
    for param in ir_extractor.extractor.parameters():
        param.requires_grad = False
    for param in fusion.parameters():
        param.requires_grad = False

    # Optimizer - only optimize temporal model (LSTM + predictor)
    trainable_params = list(model.temporal.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate)

    # Datasets
    train_dataset = NormalVideoDataset(
        train_videos,
        train_ir_videos,
        sequence_length=16,
        rgb_dropout_prob=0.15,
        ir_dropout_prob=0.15
    )

    val_dataset = NormalVideoDataset(
        val_videos,
        val_ir_videos,
        sequence_length=16,
        rgb_dropout_prob=0.0,  # No dropout during validation
        ir_dropout_prob=0.0
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_loss = float('inf')

    logger.info(f"Starting training: {num_epochs} epochs, batch_size={batch_size}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch,
                                segmentation, rgb_extractor, ir_extractor, fusion)
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, device,
                           segmentation, rgb_extractor, ir_extractor, fusion)
        logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                str(checkpoint_path / "best_model.pth"),
                epoch,
                val_loss,
                config.__dict__
            )
            logger.info(f"New best model saved (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            model.save_checkpoint(
                str(checkpoint_path / f"checkpoint_epoch_{epoch}.pth"),
                epoch,
                val_loss,
                config.__dict__
            )

    logger.info("Training complete")


def validate(model: VigiLensModel,
             dataloader: DataLoader,
             device: str,
             segmentation: YOLOSegmentation,
             rgb_extractor: RGBFeatureExtractor,
             ir_extractor: IRFeatureExtractor,
             fusion: ModalityAwareFusion) -> float:
    """
    Validate the model.

    Args:
        model: VigiLens model
        dataloader: Validation dataloader
        device: Device
        segmentation: YOLO segmentation model
        rgb_extractor: RGB feature extractor
        ir_extractor: IR feature extractor
        fusion: Modality-aware fusion module

    Returns:
        Average validation loss
    """
    model.eval()
    segmentation.model.eval()
    rgb_extractor.extractor.eval()
    ir_extractor.extractor.eval()
    fusion.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            rgb_sequence = batch['rgb_sequence']
            ir_sequence = batch['ir_sequence']
            rgb_available = batch['rgb_available']
            ir_available = batch['ir_available']

            batch_size = rgb_sequence.shape[0]
            sequence_length = rgb_sequence.shape[1]

            hidden_state = model.init_hidden(batch_size, device)

            batch_loss = 0.0
            for t in range(sequence_length - 1):
                rgb_frame = rgb_sequence[:, t]
                ir_frame = ir_sequence[:, t] if ir_sequence is not None else None

                rgb_next = rgb_sequence[:, t + 1]
                ir_next = ir_sequence[:, t + 1] if ir_sequence is not None else None

                rgb_avail_t = rgb_available[t].item() if isinstance(rgb_available, torch.Tensor) else True
                ir_avail_t = ir_available[t].item() if isinstance(ir_available, torch.Tensor) and ir_frame is not None else False

                # Extract RGB features
                rgb_features_list = []
                rgb_masks_list = []
                for b in range(batch_size):
                    if rgb_avail_t:
                        rgb_np = rgb_frame[b].cpu().numpy().astype(np.uint8)
                        rgb_mask = segmentation.extract_person_mask(rgb_np)
                        rgb_features_np = rgb_extractor.extract(rgb_np, rgb_mask)
                        rgb_features_list.append(rgb_features_np)
                        rgb_masks_list.append(rgb_mask)
                    else:
                        rgb_features_list.append(np.zeros(2048))
                        rgb_masks_list.append(None)

                # Extract IR features
                ir_features_list = []
                if ir_frame is not None:
                    for b in range(batch_size):
                        if ir_avail_t and rgb_masks_list[b] is not None:
                            ir_np = ir_frame[b].cpu().numpy().astype(np.uint8)
                            ir_mask = cv2.resize(rgb_masks_list[b], (ir_np.shape[1], ir_np.shape[0]))
                            ir_features_np = ir_extractor.extract(ir_np, ir_mask)
                            ir_features_list.append(ir_features_np)
                        elif ir_avail_t:
                            ir_np = ir_frame[b].cpu().numpy().astype(np.uint8)
                            ir_features_np = ir_extractor.extract(ir_np, None)
                            ir_features_list.append(ir_features_np)
                        else:
                            ir_features_list.append(np.zeros(2048))
                else:
                    ir_features_list = [np.zeros(2048)] * batch_size

                rgb_features = torch.from_numpy(np.array(rgb_features_list)).float().to(device)
                ir_features = torch.from_numpy(np.array(ir_features_list)).float().to(device) if any(f.any() for f in ir_features_list) else None

                fused = fusion(rgb_features, ir_features, rgb_avail_t, ir_avail_t)

                # Extract next frame features
                rgb_next_features_list = []
                rgb_next_masks_list = []
                for b in range(batch_size):
                    if rgb_avail_t:
                        rgb_next_np = rgb_next[b].cpu().numpy().astype(np.uint8)
                        rgb_next_mask = segmentation.extract_person_mask(rgb_next_np)
                        rgb_next_features_np = rgb_extractor.extract(rgb_next_np, rgb_next_mask)
                        rgb_next_features_list.append(rgb_next_features_np)
                    else:
                        rgb_next_features_list.append(np.zeros(2048))

                ir_next_features_list = []
                if ir_next is not None:
                    for b in range(batch_size):
                        if ir_avail_t and rgb_next_masks_list[b] is not None:
                            ir_next_np = ir_next[b].cpu().numpy().astype(np.uint8)
                            ir_next_mask = cv2.resize(rgb_next_masks_list[b], (ir_next_np.shape[1], ir_next_np.shape[0]))
                            ir_next_features_np = ir_extractor.extract(ir_next_np, ir_next_mask)
                            ir_next_features_list.append(ir_next_features_np)
                        elif ir_avail_t:
                            ir_next_np = ir_next[b].cpu().numpy().astype(np.uint8)
                            ir_next_features_np = ir_extractor.extract(ir_next_np, None)
                            ir_next_features_list.append(ir_next_features_np)
                        else:
                            ir_next_features_list.append(np.zeros(2048))
                else:
                    ir_next_features_list = [np.zeros(2048)] * batch_size

                rgb_next_features = torch.from_numpy(np.array(rgb_next_features_list)).float().to(device)
                ir_next_features = torch.from_numpy(np.array(ir_next_features_list)).float().to(device) if any(f.any() for f in ir_next_features_list) else None

                fused_next = fusion(rgb_next_features, ir_next_features, rgb_avail_t, ir_avail_t)

                loss, hidden_state = model.training_step(
                    fused.unsqueeze(1),
                    fused_next.unsqueeze(1),
                    hidden_state
                )

                batch_loss += loss.item()

            total_loss += batch_loss
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    config = SystemConfig()

    # In real usage, provide actual video paths
    train_videos = [Path("Data/train/rgb/video1.mp4")]
    val_videos = [Path("Data/val/rgb/video1.mp4")]

    train(
        config=config,
        train_videos=train_videos,
        val_videos=val_videos,
        num_epochs=10,
        batch_size=2
    )
