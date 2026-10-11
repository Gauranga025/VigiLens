"""
Normal-only training dataset for VigiLens.

Loads normal pedestrian videos and generates temporal sequences for next-feature prediction.
Supports modality dropout for RGB/IR missing-modality training.
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import random

logger = logging.getLogger(__name__)


class NormalVideoDataset(Dataset):
    """
    Dataset for normal pedestrian videos.

    Generates temporal sequences for next-feature prediction training.
    Splits by VIDEO to avoid data leakage (frames from same video not in train/val).
    """

    def __init__(self,
                 video_paths: List[Path],
                 ir_video_paths: Optional[List[Path]] = None,
                 sequence_length: int = 16,
                 rgb_dropout_prob: float = 0.15,
                 ir_dropout_prob: float = 0.15,
                 input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize normal video dataset.

        Args:
            video_paths: List of RGB video paths
            ir_video_paths: List of IR video paths (None if IR unavailable)
            sequence_length: Length of temporal sequences for training
            rgb_dropout_prob: Probability of dropping RGB modality
            ir_dropout_prob: Probability of dropping IR modality
            input_size: Target input size (H, W)
        """
        self.video_paths = video_paths
        self.ir_video_paths = ir_video_paths if ir_video_paths else [None] * len(video_paths)
        self.sequence_length = sequence_length
        self.rgb_dropout_prob = rgb_dropout_prob
        self.ir_dropout_prob = ir_dropout_prob
        self.input_size = input_size

        # Precompute frame counts for each video
        self.video_info = []
        for rgb_path, ir_path in zip(video_paths, self.ir_video_paths):
            cap = cv2.VideoCapture(str(rgb_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            self.video_info.append({
                'rgb_path': rgb_path,
                'ir_path': ir_path,
                'frame_count': frame_count,
                'fps': fps
            })

        logger.info(f"Dataset initialized: {len(video_paths)} videos, seq_len={sequence_length}")

    def __len__(self) -> int:
        """Return number of possible sequences."""
        total_sequences = 0
        for info in self.video_info:
            sequences_per_video = max(0, info['frame_count'] - self.sequence_length)
            total_sequences += sequences_per_video
        return total_sequences

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sequence.

        Returns:
            Dictionary containing:
            - rgb_sequence: (T, H, W, 3)
            - ir_sequence: (T, H, W) or None
            - rgb_available: bool
            - ir_available: bool
        """
        # Find which video this index belongs to
        cumulative = 0
        video_idx = 0
        frame_idx = 0
        for i, info in enumerate(self.video_info):
            if idx < cumulative + info['frame_count'] - self.sequence_length:
                video_idx = i
                frame_idx = idx - cumulative
                break
            cumulative += info['frame_count'] - self.sequence_length
        else:
            # If loop completes without break, idx is out of bounds
            # Wrap around to last video
            video_idx = len(self.video_info) - 1
            frame_idx = max(0, self.video_info[video_idx]['frame_count'] - self.sequence_length - 1)

        info = self.video_info[video_idx]

        # Load sequence from video
        rgb_sequence, ir_sequence = self._load_sequence(
            info['rgb_path'],
            info['ir_path'],
            frame_idx,
            self.sequence_length
        )

        # Apply modality dropout
        # Ensure at least one modality remains available
        rgb_available = random.random() > self.rgb_dropout_prob
        ir_available = random.random() > self.ir_dropout_prob and ir_sequence is not None

        # If both would be unavailable, force RGB to be available
        if not rgb_available and not ir_available:
            rgb_available = True

        return {
            'rgb_sequence': rgb_sequence,
            'ir_sequence': ir_sequence,
            'rgb_available': rgb_available,
            'ir_available': ir_available
        }

    def _load_sequence(self,
                      rgb_path: Path,
                      ir_path: Optional[Path],
                      start_frame: int,
                      length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a sequence of frames from video."""
        cap_rgb = cv2.VideoCapture(str(rgb_path))
        if not cap_rgb.isOpened():
            raise ValueError(f"Cannot open RGB video: {rgb_path}")
        
        cap_ir = cv2.VideoCapture(str(ir_path)) if ir_path and ir_path.exists() else None
        if cap_ir and not cap_ir.isOpened():
            logger.warning(f"Cannot open IR video: {ir_path}. Using RGB only.")
            cap_ir.release()
            cap_ir = None

        rgb_frames = []
        ir_frames = []

        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if cap_ir:
            cap_ir.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(length):
            ret_rgb, frame_rgb = cap_rgb.read()
            if not ret_rgb:
                logger.warning(f"RGB video ended early at frame {len(rgb_frames)}")
                break

            frame_rgb = cv2.resize(frame_rgb, self.input_size)
            rgb_frames.append(frame_rgb)

            if cap_ir:
                ret_ir, frame_ir = cap_ir.read()
                if ret_ir:
                    frame_ir = cv2.resize(frame_ir, self.input_size)
                    if len(frame_ir.shape) == 3:
                        frame_ir = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
                    ir_frames.append(frame_ir)
                else:
                    logger.warning(f"IR video ended early at frame {len(ir_frames)}")

        cap_rgb.release()
        if cap_ir:
            cap_ir.release()

        if len(rgb_frames) < length:
            logger.warning(f"RGB sequence shorter than requested: {len(rgb_frames)} < {length}")
        
        if ir_frames and len(ir_frames) < len(rgb_frames):
            logger.warning(f"IR sequence shorter than RGB: {len(ir_frames)} < {len(rgb_frames)}")

        rgb_sequence = np.array(rgb_frames, dtype=np.float32) / 255.0
        
        # Return zero tensor if IR unavailable (for DataLoader compatibility)
        if ir_frames:
            ir_sequence = np.array(ir_frames, dtype=np.float32) / 255.0
        else:
            # Return zero tensor with shape (length, H, W)
            ir_sequence = np.zeros((length, self.input_size[0], self.input_size[1]), dtype=np.float32)

        return rgb_sequence, ir_sequence


def split_videos_by_video(video_paths: List[Path],
                         ir_video_paths: Optional[List[Path]],
                         train_ratio: float = 0.8) -> Tuple[List[Path], List[Path], Optional[List[Path]], Optional[List[Path]]]:
    """
    Split videos by VIDEO (not by frames) to avoid data leakage.

    Args:
        video_paths: List of RGB video paths
        ir_video_paths: List of IR video paths
        train_ratio: Ratio for training split

    Returns:
        Tuple of (train_rgb, val_rgb, train_ir, val_ir)
    """
    num_videos = len(video_paths)
    num_train = int(num_videos * train_ratio)

    indices = list(range(num_videos))
    random.shuffle(indices)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_rgb = [video_paths[i] for i in train_indices]
    val_rgb = [video_paths[i] for i in val_indices]

    train_ir = [ir_video_paths[i] for i in train_indices] if ir_video_paths else None
    val_ir = [ir_video_paths[i] for i in val_indices] if ir_video_paths else None

    logger.info(f"Video split: {len(train_rgb)} train, {len(val_rgb)} val")
    return train_rgb, val_rgb, train_ir, val_ir
