"""
Frame synchronization module for visible and IR/thermal video inputs.

This module provides explicit synchronization between visible and IR frames,
handling different frame rates and ensuring proper temporal pairing.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PairedFrame:
    """
    Represents a synchronized visible-IR frame pair.
    
    Attributes:
        visible_frame: Visible RGB frame
        ir_frame: IR/thermal frame (None if unavailable)
        frame_index: Sequential frame index
        visible_timestamp: Timestamp of visible frame (if available)
        ir_timestamp: Timestamp of IR frame (if available)
    """
    visible_frame: np.ndarray
    ir_frame: Optional[np.ndarray]
    frame_index: int
    visible_timestamp: Optional[float] = None
    ir_timestamp: Optional[float] = None


class FrameSynchronizer:
    """
    Synchronizes visible and IR video frames.
    
    Handles different frame rates by using frame indices and FPS information.
    When FPS differs, uses linear interpolation or frame skipping based on configuration.
    
    Assumption: If exact timestamp metadata is unavailable, assumes videos are
    temporally aligned at the start and uses frame-index/FPS-based pairing.
    """
    
    def __init__(self,
                 visible_fps: float,
                 ir_fps: Optional[float] = None,
                 sync_tolerance: float = 0.1):
        """
        Initialize frame synchronizer.
        
        Args:
            visible_fps: Frame rate of visible video
            ir_fps: Frame rate of IR video (None if IR unavailable)
            sync_tolerance: Tolerance for timestamp matching (in seconds)
        """
        self.visible_fps = visible_fps
        self.ir_fps = ir_fps
        self.sync_tolerance = sync_tolerance
        
        self.visible_frame_index = 0
        self.ir_frame_index = 0
        
        self.ir_available = ir_fps is not None
        
        if self.ir_available:
            self.fps_ratio = visible_fps / ir_fps
            logger.info(f"Frame synchronizer: visible_fps={visible_fps:.2f}, ir_fps={ir_fps:.2f}, ratio={self.fps_ratio:.3f}")
        else:
            logger.info("Frame synchronizer: IR unavailable - visible only mode")
    
    def sync_frames(self,
                   visible_frame: np.ndarray,
                   ir_frame: Optional[np.ndarray],
                   visible_timestamp: Optional[float] = None,
                   ir_timestamp: Optional[float] = None) -> Optional[PairedFrame]:
        """
        Synchronize a visible frame with an IR frame.
        
        Args:
            visible_frame: Current visible frame
            ir_frame: Current IR frame (None if unavailable)
            visible_timestamp: Timestamp of visible frame (if available)
            ir_timestamp: Timestamp of IR frame (if available)
        
        Returns:
            PairedFrame if synchronization successful, None if end of stream
        """
        if visible_frame is None:
            return None
        
        if not self.ir_available or ir_frame is None:
            # Visible only mode
            paired = PairedFrame(
                visible_frame=visible_frame,
                ir_frame=None,
                frame_index=self.visible_frame_index,
                visible_timestamp=visible_timestamp,
                ir_timestamp=None
            )
            self.visible_frame_index += 1
            return paired
        
        # Both visible and IR available
        if visible_timestamp is not None and ir_timestamp is not None:
            # Timestamp-based synchronization
            return self._sync_by_timestamp(visible_frame, ir_frame, 
                                          visible_timestamp, ir_timestamp)
        else:
            # Frame-index/FPS-based synchronization
            return self._sync_by_index(visible_frame, ir_frame)
    
    def _sync_by_timestamp(self,
                          visible_frame: np.ndarray,
                          ir_frame: np.ndarray,
                          visible_timestamp: float,
                          ir_timestamp: float) -> Optional[PairedFrame]:
        """
        Synchronize using timestamps.
        
        This is the preferred method if timestamps are available.
        """
        time_diff = abs(visible_timestamp - ir_timestamp)
        
        if time_diff <= self.sync_tolerance:
            # Frames are temporally aligned
            paired = PairedFrame(
                visible_frame=visible_frame,
                ir_frame=ir_frame,
                frame_index=self.visible_frame_index,
                visible_timestamp=visible_timestamp,
                ir_timestamp=ir_timestamp
            )
            self.visible_frame_index += 1
            self.ir_frame_index += 1
            return paired
        else:
            # Timestamps don't match - need to skip frames
            if visible_timestamp < ir_timestamp:
                # Visible is behind, skip visible (caller should advance visible)
                logger.debug(f"Timestamp mismatch: visible behind by {time_diff:.3f}s")
                return None
            else:
                # IR is behind, skip IR (caller should advance IR)
                logger.debug(f"Timestamp mismatch: IR behind by {time_diff:.3f}s")
                return None
    
    def _sync_by_index(self,
                      visible_frame: np.ndarray,
                      ir_frame: np.ndarray) -> Optional[PairedFrame]:
        """
        Synchronize using frame indices and FPS ratio.
        
        This is a deterministic fallback when timestamps are unavailable.
        Assumes videos are temporally aligned at the start.
        """
        # Calculate expected IR frame index for current visible frame
        expected_ir_index = int(self.visible_frame_index / self.fps_ratio)
        
        if self.ir_frame_index == expected_ir_index:
            # Frames are aligned
            paired = PairedFrame(
                visible_frame=visible_frame,
                ir_frame=ir_frame,
                frame_index=self.visible_frame_index,
                visible_timestamp=None,
                ir_timestamp=None
            )
            self.visible_frame_index += 1
            self.ir_frame_index += 1
            return paired
        elif self.ir_frame_index < expected_ir_index:
            # IR is behind, skip IR frames
            # Caller should advance IR until we reach expected index
            logger.debug(f"IR behind: current={self.ir_frame_index}, expected={expected_ir_index}")
            return None
        else:
            # IR is ahead, skip visible frame
            logger.debug(f"IR ahead: current={self.ir_frame_index}, expected={expected_ir_index}")
            return None
    
    def reset(self):
        """Reset synchronizer state."""
        self.visible_frame_index = 0
        self.ir_frame_index = 0
        logger.info("Frame synchronizer reset")


class SynchronizedFrameSource:
    """
    Wrapper around frame sources that provides synchronized frame pairs.
    
    Integrates with existing frame source classes to add synchronization logic.
    """
    
    def __init__(self, frame_source, sync_tolerance: float = 0.1):
        """
        Initialize synchronized frame source.
        
        Args:
            frame_source: Base frame source (VideoFrameSource, DirectoryFrameSource, etc.)
            sync_tolerance: Tolerance for timestamp matching
        """
        self.frame_source = frame_source
        self.sync_tolerance = sync_tolerance
        
        # Get FPS information if available
        self.visible_fps = getattr(frame_source, 'visible_fps', 30.0)
        self.ir_fps = getattr(frame_source, 'ir_fps', None)
        
        # Initialize synchronizer
        self.synchronizer = FrameSynchronizer(
            visible_fps=self.visible_fps,
            ir_fps=self.ir_fps,
            sync_tolerance=sync_tolerance
        )
        
        # Buffer for IR frames when needed
        self.ir_buffer = None
        self.ir_timestamp_buffer = None
        
        logger.info("Synchronized frame source initialized")
    
    def read(self) -> Optional[PairedFrame]:
        """
        Read a synchronized frame pair.
        
        Returns:
            PairedFrame or None if end of stream
        """
        visible_frame, ir_frame = self.frame_source.read()
        
        if visible_frame is None:
            return None
        
        # Get timestamps if available (OpenCV doesn't provide them by default)
        visible_timestamp = None
        ir_timestamp = None
        
        # Attempt synchronization
        paired = self.synchronizer.sync_frames(
            visible_frame, ir_frame, visible_timestamp, ir_timestamp
        )
        
        return paired
    
    def release(self):
        """Release resources."""
        self.frame_source.release()
        self.synchronizer.reset()
