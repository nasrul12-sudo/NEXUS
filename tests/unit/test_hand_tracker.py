"""Test HandTracker dengan mock MediaPipe.

Test ini TIDAK membutuhkan mediapipe terinstall. Kita mock
dependency internal agar test bisa jalan di environment minimal.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nexus.vision.hand_tracking.base import HandTracker
from nexus.vision.hand_tracking.landmarks import NUM_LANDMARKS, HandLandmarks
from nexus.vision.hand_tracking.mediapipe_tracker import (
    HandTrackerError,
    MediaPipeHandTracker,
)


def test_abstract_base():
    with pytest.raises(TypeError):
        HandTracker()  # type: ignore[abstract]


def test_setup_missing_model(tmp_path):
    tracker = MediaPipeHandTracker(model_path=tmp_path / "nonexistent.task")
    with pytest.raises(HandTrackerError, match="not found"):
        tracker.setup()


def test_setup_invalid_running_mode(tmp_path):
    model = tmp_path / "fake.task"
    model.write_bytes(b"fake")
    tracker = MediaPipeHandTracker(
        model_path=model, running_mode="INVALID_MODE"
    )
    # mediapipe mungkin tidak terinstall — skip jika ImportError
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        pytest.skip("mediapipe not installed")
    with pytest.raises(HandTrackerError, match="Invalid running_mode"):
        tracker.setup()


def test_detect_without_setup():
    tracker = MediaPipeHandTracker()
    with pytest.raises(HandTrackerError, match="not setup"):
        tracker.detect(np.zeros((480, 640, 3), dtype=np.uint8))


def test_max_hands():
    tracker = MediaPipeHandTracker(max_hands=1)
    assert tracker.max_hands == 1
    tracker2 = MediaPipeHandTracker(max_hands=2)
    assert tracker2.max_hands == 2


def test_close_idempotent():
    tracker = MediaPipeHandTracker()
    tracker.close()
    tracker.close()  # tidak boleh raise