"""Tests for head tracker process management."""

import numpy as np
import pytest

from reachy_mini_conversation_app.vision.head_tracker import HeadTracker


def test_head_tracker_round_trip() -> None:
    """A frame can be processed by the child process."""
    tracker = HeadTracker("stub")
    try:
        frame = np.zeros((12, 20, 3), dtype=np.uint8)
        eye_center, roll = tracker.get_head_position(frame)

        assert eye_center is not None
        assert np.allclose(eye_center, np.array([0.0, 0.0], dtype=np.float32))
        assert roll == 0.0
    finally:
        tracker.close()


def test_head_tracker_rejects_unknown_backend() -> None:
    """Unknown backends fail fast with a helpful error."""
    with pytest.raises(RuntimeError, match="Unsupported head tracker backend"):
        tracker = HeadTracker("unknown")
        tracker.close()


def test_head_tracker_close_is_idempotent() -> None:
    """close() can be called repeatedly without crashing."""
    tracker = HeadTracker("stub")
    tracker.close()
    tracker.close()
