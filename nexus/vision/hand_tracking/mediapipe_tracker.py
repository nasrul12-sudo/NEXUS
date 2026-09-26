"""MediaPipe Hand Landmarker implementation.

Menggunakan MediaPipe Tasks API (bukan legacy solutions API).

Reference:
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
"""
from __future__ import annotations

import time
import cv2
from pathlib import Path
from typing import Any

import numpy as np

from nexus.core.errors import NexusError
from nexus.core.logging import get_logger
from nexus.core.metrics import get_metrics
from nexus.vision.hand_tracking.base import HandTracker
from nexus.vision.hand_tracking.landmarks import HandLandmarks


log = get_logger(__name__)


class HandTrackerError(NexusError):
    code = "HAND_TRACKER_ERROR"


class MediaPipeHandTracker(HandTracker):
    """MediaPipe Hand Landmarker wrapper.

    Args:
        model_path: path ke hand_landmarker.task. Jika None, gunakan default.
        max_hands: max number of hands (1 atau 2)
        min_detection_confidence: threshold untuk detection
        min_presence_confidence: threshold untuk presence
        min_tracking_confidence: threshold untuk tracking
        running_mode: IMAGE | VIDEO | LIVE_STREAM
    """

    #: Default model path (relatif ke project root)
    DEFAULT_MODEL_PATH = Path("data/models/hand_landmarker.task")

    def __init__(
        self,
        model_path: str | Path | None = None,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "VIDEO",
    ):
        self._model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self._max_hands = max_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_presence_confidence = min_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._running_mode = running_mode

        self._landmarker: Any = None
        self._last_timestamp_ms: int = 0

    def setup(self) -> None:
        # Import di sini supaya module ini bisa di-import tanpa mediapipe
        # (berguna untuk test yang tidak butuh mediapipe)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise HandTrackerError(
                "mediapipe not installed. Install with: pip install 'nexus[vision]'"
            ) from e

        if not self._model_path.exists():
            raise HandTrackerError(
                f"Model file not found: {self._model_path}. "
                f"Download dari https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        running_mode_map = {
            "IMAGE": mp_vision.RunningMode.IMAGE,
            "VIDEO": mp_vision.RunningMode.VIDEO,
            "LIVE_STREAM": mp_vision.RunningMode.LIVE_STREAM,
        }
        if self._running_mode not in running_mode_map:
            raise HandTrackerError(
                f"Invalid running_mode: {self._running_mode}. "
                f"Must be one of {list(running_mode_map.keys())}"
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(self._model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode_map[self._running_mode],
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._min_detection_confidence,
            min_hand_presence_confidence=self._min_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        log.info(
            "hand_tracker.setup",
            model_path=str(self._model_path),
            max_hands=self._max_hands,
            running_mode=self._running_mode,
        )

    def detect(self, image: np.ndarray) -> list[HandLandmarks]:
        if self._landmarker is None:
            raise HandTrackerError("HandTracker not setup. Call setup() first.")

        try:
            import mediapipe as mp
        except ImportError as e:
            raise HandTrackerError("mediapipe not installed") from e

        metrics = get_metrics()
        with metrics.timer("vision.hand_tracker.detect_ms"):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            timestamp_ms = self._next_timestamp_ms()
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands: list[HandLandmarks] = []
        if not result.hand_landmarks:
            return hands

        for i, hand_landmarks in enumerate(result.hand_landmarks):
            # hand_landmarks: list of 21 NormalizedLandmark (x, y, z)
            landmarks_arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32,
            )

            # Handedness
            handedness = "Right"  # default
            confidence = 0.0
            if result.handedness and i < len(result.handedness):
                h = result.handedness[i]
                if h:
                    handedness = h[0].category_name  # "Left" or "Right"
                    confidence = float(h[0].score)

            # World landmarks (metric 3D)
            world_arr = None
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                wl = result.hand_world_landmarks[i]
                world_arr = np.array(
                    [[lm.x, lm.y, lm.z] for lm in wl],
                    dtype=np.float32,
                )

            hands.append(HandLandmarks(
                landmarks=landmarks_arr,
                handedness=handedness,
                confidence=confidence,
                world_landmarks=world_arr,
            ))

        MIN_HAND_CONF = 0.5
        hands = [h for h in hands if h.confidence >= MIN_HAND_CONF]

        return hands

    def _next_timestamp_ms(self) -> int:
        """Generate monotonically increasing timestamp untuk MediaPipe VIDEO mode."""
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
                log.exception("hand_tracker.close_error")
            self._landmarker = None
            log.info("hand_tracker.closed")

    @property
    def max_hands(self) -> int:
        return self._max_hands