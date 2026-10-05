"""
YOLO segmentation module for pedestrian ROI extraction.

Uses YOLOv8n-seg for lightweight instance segmentation.
Extracts pedestrian masks for feature extraction ROI.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOSegmentation:
    """
    YOLO segmentation wrapper for pedestrian ROI extraction.

    Uses YOLOv8n-seg (segmentation variant) for instance segmentation.
    Focuses on the 'person' class for pedestrian detection.
    """

    def __init__(self, model_path: str = "yolov8n-seg.pt", device: str = "cuda"):
        """
        Initialize YOLO segmentation model.

        Args:
            model_path: Path to YOLO segmentation model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        self.person_class_id = 0  # COCO 'person' class ID

        logger.info(f"YOLO segmentation loaded: {model_path} on {device}")

    def extract_person_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pedestrian mask from frame.

        Policy: Use the largest/highest-confidence pedestrian ROI.
        If multiple people detected, combine their masks.

        Args:
            frame: Input RGB frame (H, W, 3)

        Returns:
            Binary mask (H, W) where 1 = pedestrian region, 0 = background
            Returns None if no person detected.
        """
        results = self.model(frame, verbose=False)

        if results is None or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
            return None

        # Filter for person class
        person_indices = [
            i for i, cls in enumerate(results[0].boxes.cls)
            if int(cls) == self.person_class_id
        ]

        if not person_indices:
            return None

        # Get masks for detected persons
        masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
        confidences = results[0].boxes.conf.cpu().numpy()

        # Filter to person masks
        person_masks = masks[person_indices]
        person_confs = confidences[person_indices]

        # Policy: Combine all person masks into one ROI
        # This handles multiple pedestrians by including all
        combined_mask = np.zeros(person_masks.shape[1:], dtype=np.uint8)

        for mask in person_masks:
            mask_binary = (mask > 0.5).astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask_binary)

        return combined_mask

    def extract_largest_person_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract mask for the largest detected pedestrian.

        Alternative policy: Use only the largest pedestrian ROI.

        Args:
            frame: Input RGB frame (H, W, 3)

        Returns:
            Binary mask (H, W) for largest pedestrian
            Returns None if no person detected.
        """
        results = self.model(frame, verbose=False)

        if results is None or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
            return None

        person_indices = [
            i for i, cls in enumerate(results[0].boxes.cls)
            if int(cls) == self.person_class_id
        ]

        if not person_indices:
            return None

        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Find largest box by area
        largest_idx = None
        max_area = 0

        for idx in person_indices:
            x1, y1, x2, y2 = boxes[idx]
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_idx = idx

        if largest_idx is None:
            return None

        mask = masks[largest_idx]
        mask_binary = (mask > 0.5).astype(np.uint8)

        return mask_binary

    def apply_mask_to_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply binary mask to frame, zeroing out background.

        Args:
            frame: Input frame (H, W, 3)
            mask: Binary mask (H, W)

        Returns:
            Masked frame (H, W, 3)
        """
        mask_expanded = mask[:, :, np.newaxis]
        masked = frame * mask_expanded
        return masked.astype(np.uint8)

    def get_masked_roi(self, frame: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Get masked frame and bounding box of ROI.

        Args:
            frame: Input frame (H, W, 3)
            mask: Binary mask (H, W)

        Returns:
            Tuple of (masked_frame, bbox) where bbox is (x1, y1, x2, y2)
        """
        # Get bounding box of mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return frame, (0, 0, frame.shape[1], frame.shape[0])

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Apply mask
        masked = self.apply_mask_to_frame(frame, mask)

        return masked, (int(x1), int(y1), int(x2), int(y2))
