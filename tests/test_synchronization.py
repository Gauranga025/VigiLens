"""
Tests for frame synchronization module.
"""

import numpy as np
from pipeline.synchronization import FrameSynchronizer


def test_equal_fps_synchronization():
    """Test synchronization with equal FPS streams."""
    sync = FrameSynchronizer(visible_fps=30.0, ir_fps=30.0)

    # Simulate equal FPS
    visible_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ir_frame = np.zeros((480, 640, 1), dtype=np.uint8)

    paired = sync.sync_frames(visible_frame, ir_frame)

    assert paired is not None
    assert paired.frame_index == 0
    assert paired.visible_frame is not None
    assert paired.ir_frame is not None


def test_different_fps_synchronization():
    """Test synchronization with different FPS streams."""
    sync = FrameSynchronizer(visible_fps=30.0, ir_fps=15.0)

    visible_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ir_frame = np.zeros((480, 640, 1), dtype=np.uint8)

    # First frame should pair
    paired = sync.sync_frames(visible_frame, ir_frame)
    assert paired is not None
    assert paired.frame_index == 0


def test_ir_unavailable():
    """Test synchronization when IR is unavailable."""
    sync = FrameSynchronizer(visible_fps=30.0, ir_fps=None)

    visible_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    paired = sync.sync_frames(visible_frame, None)

    assert paired is not None
    assert paired.frame_index == 0
    assert paired.ir_frame is None


def test_end_of_stream():
    """Test synchronization at end of stream."""
    sync = FrameSynchronizer(visible_fps=30.0, ir_fps=30.0)

    # None frame indicates end of stream
    paired = sync.sync_frames(None, None)

    assert paired is None


def test_reset():
    """Test synchronizer reset."""
    sync = FrameSynchronizer(visible_fps=30.0, ir_fps=30.0)

    visible_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ir_frame = np.zeros((480, 640, 1), dtype=np.uint8)

    # Process one frame
    sync.sync_frames(visible_frame, ir_frame)
    assert sync.visible_frame_index == 1

    # Reset
    sync.reset()
    assert sync.visible_frame_index == 0
    assert sync.ir_frame_index == 0


if __name__ == "__main__":
    test_equal_fps_synchronization()
    test_different_fps_synchronization()
    test_ir_unavailable()
    test_end_of_stream()
    test_reset()
    print("All synchronization tests passed!")
