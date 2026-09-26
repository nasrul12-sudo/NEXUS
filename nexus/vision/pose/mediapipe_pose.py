"""MediaPipe Pose Landmarker implementation.

Menggunakan MediaPipe Tasks API.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nexus.core.errors import NexusError
from nexus.core.logging import get_logger
from nexus.core.metrics import get_metrics
from nexus.vision.pose.base import PoseTracker
from nexus.vision.pose.landmarks import PoseLandmarks


log = get_logger(__name__)


class PoseTrackerError(NexusError):
    code = "POSE_TRACKER_ERROR"


class MediaPipePoseTracker(PoseTracker):
    """MediaPipe Pose Landmarker wrapper.

    Args:
        model_path: path ke pose_landmarker.task
        num_poses: max number of poses (biasanya 1)
        min_detection_confidence: threshold detection
        min_presence_confidence: threshold presence
        min_tracking_confidence: threshold tracking
        running_mode: IMAGE | VIDEO | LIVE_STREAM
    """

    DEFAULT_MODEL_PATH = Path("data/models/pose_landmarker_lite.task")

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "VIDEO",
    ):
        self._model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self._num_poses = num_poses
        self._min_detection_confidence = min_detection_confidence
        self._min_presence_confidence = min_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._running_mode = running_mode

        self._landmarker: Any = None
        self._last_timestamp_ms: int = 0

    def setup(self) -> None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise PoseTrackerError(
                "mediapipe not installed. Install with: pip install mediapipe"
            ) from e

        if not self._model_path.exists():
            raise PoseTrackerError(
                f"Model file not found: {self._model_path}. "
                f"Download dari https://storage.googleapis.com/mediapipe-models/"
                f"pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            )

        mode_map = {
            "IMAGE": mp_vision.RunningMode.IMAGE,
            "VIDEO": mp_vision.RunningMode.VIDEO,
            "LIVE_STREAM": mp_vision.RunningMode.LIVE_STREAM,
        }
        if self._running_mode not in mode_map:
            raise PoseTrackerError(
                f"Invalid running_mode: {self._running_mode}. "
                f"Must be one of {list(mode_map.keys())}"
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(self._model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_map[self._running_mode],
            num_poses=self._num_poses,
            min_pose_detection_confidence=self._min_detection_confidence,
            min_pose_presence_confidence=self._min_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        log.info(
            "pose_tracker.setup",
            model_path=str(self._model_path),
            num_poses=self._num_poses,
            running_mode=self._running_mode,
        )

    def detect(self, image: np.ndarray) -> PoseLandmarks | None:
        if self._landmarker is None:
            raise PoseTrackerError("PoseTracker not setup. Call setup() first.")

        try:
            import mediapipe as mp
        except ImportError as e:
            raise PoseTrackerError("mediapipe not installed") from e

        metrics = get_metrics()
        with metrics.timer("vision.pose.detect_ms"):
            # Convert BGR → RGB (MediaPipe expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            timestamp_ms = self._next_timestamp_ms()
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        # Ambil pose pertama (index 0)
        pose_lms = result.pose_landmarks[0]

        landmarks_arr = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_lms],
            dtype=np.float32,
        )
        visibility_arr = np.array(
            [lm.visibility for lm in pose_lms],
            dtype=np.float32,
        )

        world_arr = None
        if result.pose_world_landmarks:
            wl = result.pose_world_landmarks[0]
            world_arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in wl],
                dtype=np.float32,
            )

        return PoseLandmarks(
            landmarks=landmarks_arr,
            visibility=visibility_arr,
            world_landmarks=world_arr,
        )

    def _next_timestamp_ms(self) -> int:
        now_ms = int(time.time() * 1000)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms
        return now_ms

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                log.exception("pose_tracker.close_error")
            self._landmarker = None
            log.info("pose_tracker.closed")