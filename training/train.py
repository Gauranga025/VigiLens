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
from typing import Dict, List, Optional, Tuple
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

            # Modality availability is per-batch-sample, not per-timestep
            # rgb_available and ir_available are shape (B,)
            rgb_avail = rgb_available if isinstance(rgb_available, torch.Tensor) else torch.ones(batch_size, dtype=torch.bool, device=device)
            ir_avail = ir_available if isinstance(ir_available, torch.Tensor) else torch.zeros(batch_size, dtype=torch.bool, device=device)

            # Extract features for current frame with no_grad for frozen components
            with torch.no_grad():
                # Extract RGB features for current frame
                rgb_features_list = []
                rgb_masks_list = []
                for b in range(batch_size):
                    if rgb_avail[b]:
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
                # Only extract IR features if IR videos are actually available
                # (not just zero tensors from RGB-only mode)
                if ir_frame is not None and not (ir_frame == 0).all():
                    for b in range(batch_size):
                        if ir_avail[b] and rgb_masks_list[b] is not None:
                            ir_np = ir_frame[b].cpu().numpy().astype(np.uint8)
                            # Resize RGB mask to IR dimensions
                            ir_mask = cv2.resize(rgb_masks_list[b], (ir_np.shape[1], ir_np.shape[0]))
                            ir_features_np = ir_extractor.extract(ir_np, ir_mask)
                            ir_features_list.append(ir_features_np)
                            ir_masks_list.append(ir_mask)
                        elif ir_avail[b]:
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
            # Use per-sample availability for fusion
            rgb_avail_any = rgb_avail.any().item()
            ir_avail_any = ir_avail.any().item()
            fused = fusion(rgb_features, ir_features, rgb_avail_any, ir_avail_any)

            # Extract features for next frame (target) with no_grad
            with torch.no_grad():
                rgb_next_features_list = []
                rgb_next_masks_list = []
                for b in range(batch_size):
                    if rgb_avail[b]:
                        rgb_next_np = rgb_next[b].cpu().numpy().astype(np.uint8)
                        rgb_next_mask = segmentation.extract_person_mask(rgb_next_np)
                        rgb_next_features_np = rgb_extractor.extract(rgb_next_np, rgb_next_mask)
                        rgb_next_features_list.append(rgb_next_features_np)
                        rgb_next_masks_list.append(rgb_next_mask)
                    else:
                        rgb_next_features_list.append(np.zeros(2048))
                        rgb_next_masks_list.append(None)

                ir_next_features_list = []
                if ir_next is not None:
                    for b in range(batch_size):
                        if ir_avail[b] and rgb_next_masks_list[b] is not None:
                            ir_next_np = ir_next[b].cpu().numpy().astype(np.uint8)
                            ir_next_mask = cv2.resize(rgb_next_masks_list[b], (ir_next_np.shape[1], ir_next_np.shape[0]))
                            ir_next_features_np = ir_extractor.extract(ir_next_np, ir_next_mask)
                            ir_next_features_list.append(ir_next_features_np)
                        elif ir_avail[b]:
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
            fused_next = fusion(rgb_next_features, ir_next_features, rgb_avail_any, ir_avail_any)

            # Training step: z_t -> predict z_(t+1)
            loss, hidden_state = model.training_step(
                fused,  # (B, 512)
                fused_next,  # (B, 512)
                hidden_state
            )

            batch_loss += loss

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
            # Use sanity checkpoint name for 2-epoch runs
            if num_epochs == 2:
                checkpoint_name = "vigilens_sanity_epoch_2.pth"
            else:
                checkpoint_name = "best_model.pth"
            model.save_checkpoint(
                str(checkpoint_path / checkpoint_name),
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


def get_repo_root() -> Path:
    """Get repository root from the location of this script."""
    return Path(__file__).parent.parent


def discover_videos(data_dir: Path, split: str, modality: str, extensions: List[str] = None, required: bool = True) -> List[Path]:
    """
    Discover video files in dataset directory.

    Args:
        data_dir: Base data directory
        split: 'train' or 'val'
        modality: 'rgb' or 'ir'
        extensions: List of video extensions to search (default from config)
        required: If True, raises error when directory or videos not found

    Returns:
        Sorted list of video paths (empty if not required and not found)
    """
    video_dir = data_dir / split / modality
    if not video_dir.exists():
        if required:
            raise ValueError(f"Video directory not found: {video_dir}")
        else:
            logger.warning(f"Video directory not found: {video_dir}. {modality.upper()} modality will be disabled.")
            return []

    if extensions is None:
        from config.config import SystemConfig
        config = SystemConfig()
        extensions = config.training.supported_video_extensions

    videos = []
    for ext in extensions:
        videos.extend(video_dir.glob(f"*{ext}"))

    if not videos:
        if required:
            raise ValueError(f"No videos found in {video_dir} with extensions {extensions}")
        else:
            logger.warning(f"No videos found in {video_dir} with extensions {extensions}. {modality.upper()} modality will be disabled.")
            return []

    return sorted(videos)


def extract_sequence_id(video_path: Path) -> str:
    """
    Extract sequence identifier from video filename.
    
    Removes modality suffix and extension to get the base sequence ID.
    Examples:
        scene001_rgb.mp4 -> scene001
        scene001_ir.avi -> scene001
        video1_rgb.mov -> video1
    
    Args:
        video_path: Path to video file
        
    Returns:
        Sequence identifier string
    """
    stem = video_path.stem  # filename without extension
    
    # Remove common modality suffixes
    for suffix in ['_rgb', '_ir', '-rgb', '-ir', '_visible', '_thermal']:
        if stem.lower().endswith(suffix.lower()):
            stem = stem[:-len(suffix)]
    
    return stem


def pair_videos_by_sequence_id(rgb_videos: List[Path], ir_videos: List[Path]) -> List[Tuple[Path, Path]]:
    """
    Pair RGB and IR videos by their sequence identifier.
    
    Args:
        rgb_videos: List of RGB video paths
        ir_videos: List of IR video paths
        
    Returns:
        List of (rgb_path, ir_path) tuples
        
    Raises:
        ValueError: If pairing fails (unmatched videos, duplicates, etc.)
    """
    # Build dictionaries mapping sequence ID to video path
    rgb_dict = {}
    for rgb_path in rgb_videos:
        seq_id = extract_sequence_id(rgb_path)
        if seq_id in rgb_dict:
            raise ValueError(f"Duplicate RGB sequence ID: {seq_id} ({rgb_dict[seq_id]} and {rgb_path})")
        rgb_dict[seq_id] = rgb_path
    
    ir_dict = {}
    for ir_path in ir_videos:
        seq_id = extract_sequence_id(ir_path)
        if seq_id in ir_dict:
            raise ValueError(f"Duplicate IR sequence ID: {seq_id} ({ir_dict[seq_id]} and {ir_path})")
        ir_dict[seq_id] = ir_path
    
    # Find common sequence IDs
    common_ids = set(rgb_dict.keys()) & set(ir_dict.keys())
    
    # Check for unmatched videos
    unmatched_rgb = set(rgb_dict.keys()) - common_ids
    unmatched_ir = set(ir_dict.keys()) - common_ids
    
    if unmatched_rgb:
        logger.warning(f"Unmatched RGB videos: {unmatched_rgb}")
    if unmatched_ir:
        logger.warning(f"Unmatched IR videos: {unmatched_ir}")
    
    if not common_ids:
        raise ValueError("No matching RGB/IR pairs found. Check filename conventions.")
    
    # Create paired list sorted by sequence ID
    pairs = [(rgb_dict[seq_id], ir_dict[seq_id]) for seq_id in sorted(common_ids)]
    
    logger.info(f"Paired {len(pairs)} RGB/IR video pairs by sequence ID")
    
    return pairs


def validate_dataset(train_rgb: List[Path],
                    train_ir: List[Path],
                    val_rgb: List[Path],
                    val_ir: List[Path]) -> None:
    """
    Validate dataset structure and RGB/IR pairing.

    Args:
        train_rgb: Training RGB video paths
        train_ir: Training IR video paths
        val_rgb: Validation RGB video paths
        val_ir: Validation IR video paths

    Raises:
        ValueError: If validation fails
    """
    # Pair by sequence ID
    train_pairs = pair_videos_by_sequence_id(train_rgb, train_ir)
    val_pairs = pair_videos_by_sequence_id(val_rgb, val_ir)
    
    # Check videos are readable
    def check_readable(videos: List[Path], name: str):
        for video in videos:
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise ValueError(f"Cannot open {name} video: {video}")
            cap.release()

    check_readable(train_rgb, "train RGB")
    check_readable(train_ir, "train IR")
    check_readable(val_rgb, "validation RGB")
    check_readable(val_ir, "validation IR")

    # Check FPS and frame count compatibility for paired videos
    def check_compatibility(pairs: List[Tuple[Path, Path]], split_name: str):
        for rgb_path, ir_path in pairs:
            cap_rgb = cv2.VideoCapture(str(rgb_path))
            cap_ir = cv2.VideoCapture(str(ir_path))

            rgb_fps = cap_rgb.get(cv2.CAP_PROP_FPS)
            ir_fps = cap_ir.get(cv2.CAP_PROP_FPS)
            rgb_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
            ir_frames = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))

            cap_rgb.release()
            cap_ir.release()

            if abs(rgb_fps - ir_fps) > 0.1:
                logger.warning(f"{split_name}: FPS mismatch - {rgb_path.name} ({rgb_fps:.2f} fps) vs {ir_path.name} ({ir_fps:.2f} fps)")

            if rgb_frames != ir_frames:
                logger.warning(f"{split_name}: Frame count mismatch - {rgb_path.name} ({rgb_frames} frames) vs {ir_path.name} ({ir_frames} frames)")

    check_compatibility(train_pairs, "train")
    check_compatibility(val_pairs, "validation")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Get repository root
    repo_root = get_repo_root()
    data_dir = repo_root / "Data"

    # Discover videos (IR is optional)
    train_rgb_videos = discover_videos(data_dir, "train", "rgb", required=True)
    train_ir_videos = discover_videos(data_dir, "train", "ir", required=False)
    val_rgb_videos = discover_videos(data_dir, "val", "rgb", required=True)
    val_ir_videos = discover_videos(data_dir, "val", "ir", required=False)

    # Check if IR is available
    ir_available = len(train_ir_videos) > 0 or len(val_ir_videos) > 0
    
    if ir_available:
        # Validate dataset and get paired videos
        validate_dataset(train_rgb_videos, train_ir_videos, val_rgb_videos, val_ir_videos)
        
        # Re-order videos by sequence ID pairing
        train_pairs = pair_videos_by_sequence_id(train_rgb_videos, train_ir_videos)
        val_pairs = pair_videos_by_sequence_id(val_rgb_videos, val_ir_videos)
        
        train_rgb_videos = [p[0] for p in train_pairs]
        train_ir_videos = [p[1] for p in train_pairs]
        val_rgb_videos = [p[0] for p in val_pairs]
        val_ir_videos = [p[1] for p in val_pairs]
        
        logger.info(f"Training with RGB+IR: {len(train_rgb_videos)} train pairs, {len(val_rgb_videos)} val pairs")
    else:
        logger.warning("No IR videos found. Training in RGB-only mode.")
        logger.warning("RGB-only mode uses the fusion layer's single-modality adaptation.")
        logger.warning("Note: RGB-only checkpoints may not be directly comparable to RGB+IR checkpoints.")
        train_ir_videos = None
        val_ir_videos = None

    # Load config
    config = SystemConfig()

    # Override config for sanity run
    config.training.num_epochs = 2
    config.training.batch_size = 2

    # Print training summary
    print("\n" + "="*50)
    print("VigiLens Training")
    print("="*50)
    print(f"Train RGB videos: {len(train_rgb_videos)}")
    print(f"Train IR videos: {len(train_ir_videos) if train_ir_videos else 0}")
    print(f"Validation RGB videos: {len(val_rgb_videos)}")
    print(f"Validation IR videos: {len(val_ir_videos) if val_ir_videos else 0}")
    print(f"Mode: {'RGB+IR' if ir_available else 'RGB-only'}")
    print(f"\nSequence length: {config.training.sequence_length}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Learning rate: {config.training.learning_rate}")
    print("="*50 + "\n")

    # Train
    train(
        config=config,
        train_videos=train_rgb_videos,
        val_videos=val_rgb_videos,
        train_ir_videos=train_ir_videos,
        val_ir_videos=val_ir_videos,
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        checkpoint_dir="checkpoints"
    )
