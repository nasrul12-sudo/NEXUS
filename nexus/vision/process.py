"""Vision Process — camera + hand + pose + gesture."""
from __future__ import annotations

import time

from nexus.core.config import get_config
from nexus.core.events import EventType
from nexus.core.logging import get_logger
from nexus.core.metrics import get_metrics
from nexus.core.process import NexusProcess
from nexus.vision.camera import CameraProvider, OpenCVCameraProvider
from nexus.vision.context import BodyFrameExtractor
from nexus.vision.gesture import GestureEngine, make_gesture_event
from nexus.vision.hand_tracking import HandTracker, MediaPipeHandTracker
from nexus.vision.pose import MediaPipePoseTracker, PoseTracker


log = get_logger(__name__)


class VisionProcess(NexusProcess):
    """Vision process dengan pose estimation."""

    name = "vision"
    source = "vision"

    def __init__(
        self,
        camera: CameraProvider | None = None,
        hand_tracker: HandTracker | None = None,
        pose_tracker: PoseTracker | None = None,
        gesture_engine: GestureEngine | None = None,
    ):
        super().__init__()
        self._camera = camera
        self._hand_tracker = hand_tracker
        self._pose_tracker = pose_tracker
        self._gesture_engine = gesture_engine

        self._body_frame_extractor = BodyFrameExtractor()

        self._had_hand_last_frame = False
        self._fps_window: list[float] = []
        self._frames_processed = 0
        self._frames_dropped = 0
        self._gestures_detected = 0
        self._metrics = get_metrics()
        self._prev_handedness: set[str] = set()

    def setup(self) -> None:
        cfg = get_config()
        vision_cfg = cfg.vision

        # Camera
        if self._camera is None:
            self._camera = OpenCVCameraProvider(
                device=vision_cfg.camera.device,
                width=vision_cfg.camera.width,
                height=vision_cfg.camera.height,
                fps=vision_cfg.camera.fps,
                buffer_size=vision_cfg.camera.buffer_size,
            )
        self._camera.open()

        # Hand tracker
        if self._hand_tracker is None:
            self._hand_tracker = MediaPipeHandTracker(
                model_path=vision_cfg.hand_tracking.model_path,
                max_hands=vision_cfg.hand_tracking.max_hands,
                min_detection_confidence=vision_cfg.hand_tracking.min_detection_confidence,
                min_presence_confidence=vision_cfg.hand_tracking.min_presence_confidence,
                min_tracking_confidence=vision_cfg.hand_tracking.min_tracking_confidence,
                running_mode=vision_cfg.hand_tracking.running_mode,
            )
        self._hand_tracker.setup()

        # Pose tracker
        pose_cfg = getattr(vision_cfg, 'pose_tracking', None)
        pose_enabled = getattr(pose_cfg, 'enabled', True) if pose_cfg else True
        if pose_enabled:
            if self._pose_tracker is None:
                model_path = getattr(pose_cfg, 'model_path', 'data/models/pose_landmarker_lite.task') if pose_cfg else 'data/models/pose_landmarker_lite.task'
                self._pose_tracker = MediaPipePoseTracker(
                    model_path=model_path,
                    num_poses=1,
                )
            try:
                self._pose_tracker.setup()
            except Exception:
                self.log.exception("vision.pose_setup_failed")
                self._pose_tracker = None

        # Gesture engine
        if self._gesture_engine is None:
            self._gesture_engine = GestureEngine()
        self._gesture_engine.setup()

        self.log.info(
            "vision.setup_done",
            camera_fps=self._camera.fps,
            max_hands=self._hand_tracker.max_hands,
            pose_enabled=self._pose_tracker is not None,
            gesture_enabled=cfg.gesture.enabled,
        )

    def run(self) -> None:
        target_fps = self._camera.fps
        frame_interval = 1.0 / target_fps
        next_frame_time = time.perf_counter()

        while not self.stop_event.is_set():
            frame = self._camera.read()
            if frame is None:
                self._frames_dropped += 1
                time.sleep(0.005)
                continue

            # Detect hands
            try:
                hands = self._hand_tracker.detect(frame.data)
            except Exception:
                self.log.exception("vision.detect_error")
                hands = []

            # Detect pose (opsional)
            body_frame = None
            if self._pose_tracker is not None:
                try:
                    pose = self._pose_tracker.detect(frame.data)
                    if pose is not None:
                        body_frame = self._body_frame_extractor.extract(pose)
                        # Publish pose event periodically
                        if self._frames_processed % 10 == 0:
                            self.emit(
                                "pose.detected",
                                {
                                    "visibility": float(pose.visibility.mean()),
                                    "body_frame_available": body_frame is not None,
                                },
                            )
                except Exception:
                    self.log.exception("vision.pose_detect_error")

            # Set body frame ke gesture engine
            if self._gesture_engine is not None:
                self._gesture_engine.set_body_frame(body_frame)

            # Process hands
            if hands:
                self._process_hands(hands, frame.timestamp, frame.frame_index, body_frame)
                self._had_hand_last_frame = True
            elif self._had_hand_last_frame:
                self._publish_hand_lost(frame.timestamp)
                if self._gesture_engine is not None:
                    self._gesture_engine.process_hand_lost(frame.timestamp)
                self._had_hand_last_frame = False

            self._frames_processed += 1
            self._update_fps_stats(frame.timestamp)

            # Frame pacing
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.perf_counter()

            if self._frames_processed % 300 == 0:
                self._publish_metrics()

    def teardown(self) -> None:
        if self._camera is not None:
            self._camera.close()
        if self._hand_tracker is not None:
            self._hand_tracker.close()
        if self._pose_tracker is not None:
            self._pose_tracker.close()
        self.log.info(
            "vision.teardown",
            frames_processed=self._frames_processed,
            frames_dropped=self._frames_dropped,
            gestures_detected=self._gestures_detected,
        )

    def _process_hands(self, hands, timestamp, frame_index, body_frame):
        """Process hands — track mana yang masih ada."""
        self._publish_landmarks(hands, timestamp, frame_index)

        if self._gesture_engine is None:
            return

        # Track handedness yang terdeteksi frame ini
        current_handedness = {h.handedness for h in hands}

        # Cek tangan yang hilang sejak frame sebelumnya
        if hasattr(self, '_prev_handedness'):
            lost = self._prev_handedness - current_handedness
            for h in lost:
                self._gesture_engine.process_hand_lost(timestamp, handedness=h)
                log.debug("vision.hand_lost_individual", handedness=h)

        # Process tangan yang ada
        for hand in hands:
            try:
                smoothed_list = self._gesture_engine.process_hand(hand, timestamp)
            except Exception:
                self.log.exception("vision.gesture_error")
                continue

            for smoothed in smoothed_list:
                self._gestures_detected += 1
                self.publish(make_gesture_event(smoothed, hand, body_frame))

        self._prev_handedness = current_handedness

    def _publish_landmarks(self, hands, timestamp, frame_index) -> None:
        payload = {
            "frame_index": frame_index,
            "frame_timestamp": timestamp,
            "hands": [h.to_dict() for h in hands],
            "hand_count": len(hands),
        }
        self.emit(EventType.HAND_LANDMARKS, payload)

    def _publish_hand_lost(self, timestamp) -> None:
        self.emit(EventType.HAND_LOST, {"frame_timestamp": timestamp})

    def _publish_metrics(self) -> None:
        if len(self._fps_window) < 2:
            return
        elapsed = self._fps_window[-1] - self._fps_window[0]
        fps = (len(self._fps_window) - 1) / elapsed if elapsed > 0 else 0.0
        self.emit(
            "vision.fps",
            {
                "fps": round(fps, 2),
                "frames_processed": self._frames_processed,
                "frames_dropped": self._frames_dropped,
                "gestures_detected": self._gestures_detected,
            },
        )

    def _update_fps_stats(self, timestamp) -> None:
        self._fps_window.append(timestamp)
        cutoff = timestamp - 60.0
        while self._fps_window and self._fps_window[0] < cutoff:
            self._fps_window.pop(0)