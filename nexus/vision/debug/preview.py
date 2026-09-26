"""Camera preview untuk debugging — dengan hand tracking, pose estimation,
dan gesture recognition.

Menampilkan:
  - Live camera feed
  - Hand skeleton overlay (21 landmarks)
  - Pose skeleton overlay (33 landmarks)
  - Body frame axes (forward/right/up)
  - Gesture recognition (dengan body-aware features)
  - HUD: FPS, frame count, hand count, resolution, pose status
  - Snapshot (tekan 's')
  - Recording (tekan 'r')
  - Quit (tekan 'q' atau ESC)

Design:
  - Menggunakan CameraProvider, HandTracker, PoseTracker dari NEXUS
  - Stateless rendering — setiap frame di-render dari scratch
  - Configurable via PreviewConfig
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nexus.core.logging import get_logger
from nexus.vision.camera import CameraProvider, Frame
from nexus.vision.context import BodyFrameExtractor
from nexus.vision.debug.overlay import (
    draw_body_frame,
    draw_center_crosshair,
    draw_feature_debug,
    draw_fps_indicator,
    draw_hand_label,
    draw_hand_skeleton,
    draw_hud_text,
    draw_pose_skeleton,
)
from nexus.vision.gesture import Gesture, GestureEngine
from nexus.vision.hand_tracking import HandTracker
from nexus.vision.pose import PoseTracker


log = get_logger(__name__)


# ---------- Config ----------

@dataclass
class PreviewConfig:
    """Konfigurasi preview."""

    window_name: str = "NEXUS Camera Preview"
    window_width: int = 960
    window_height: int = 540

    # Display toggles
    show_skeleton: bool = True
    show_labels: bool = True
    show_hud: bool = True
    show_crosshair: bool = False
    show_landmark_names: bool = False
    show_gesture: bool = True
    show_features: bool = False
    show_pose: bool = True
    show_body_frame: bool = True

    mirror: bool = True
    fps_window: int = 30

    # Output dirs
    snapshot_dir: Path = field(default_factory=lambda: Path("data/snapshots"))
    recording_dir: Path = field(default_factory=lambda: Path("data/recordings"))
    recording_fps: float = 30.0


# ---------- State per hand ----------

@dataclass
class _HandGestureState:
    gesture: Gesture = Gesture.UNKNOWN
    confidence: float = 0.0
    timestamp: float = 0.0


# ---------- Preview ----------

class CameraPreview:
    """Camera preview dengan hand tracking + pose + gesture recognition."""

    def __init__(
        self,
        camera: CameraProvider,
        hand_tracker: HandTracker,
        config: PreviewConfig | None = None,
        gesture_engine: GestureEngine | None = None,
        pose_tracker: PoseTracker | None = None,
    ):
        self._camera = camera
        self._hand_tracker = hand_tracker
        self._pose_tracker = pose_tracker
        self._config = config or PreviewConfig()
        self._gesture_engine = gesture_engine or GestureEngine()
        self._body_frame_extractor = BodyFrameExtractor()

        # State
        self._frame_count = 0
        self._start_time = 0.0
        self._frame_timestamps: list[float] = []
        self._last_hands: list = []
        self._last_pose: Any = None
        self._last_body_frame: Any = None
        self._detection_ms_samples: list[float] = []
        self._pose_ms_samples: list[float] = []
        self._last_gestures: dict[str, _HandGestureState] = {}

        # Recording
        self._recording = False
        self._video_writer: cv2.VideoWriter | None = None
        self._recording_path: Path | None = None

        # Ensure dirs
        self._config.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._config.recording_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Lifecycle ----------

    def run(self) -> None:
        """Main loop. Blocks sampai user quit."""
        self._setup_window()
        self._start_time = time.perf_counter()
        self._gesture_engine.setup()

        log.info(
            "preview.start",
            window=self._config.window_name,
            pose_enabled=self._pose_tracker is not None,
        )

        try:
            while True:
                frame = self._camera.read()
                if frame is None:
                    log.warning("preview.empty_frame")
                    time.sleep(0.01)
                    continue

                # ---- Hand detection ----
                t0 = time.perf_counter()
                try:
                    hands = self._hand_tracker.detect(frame.data)
                except Exception:
                    log.exception("preview.detect_error")
                    hands = []
                detection_ms = (time.perf_counter() - t0) * 1000
                self._detection_ms_samples.append(detection_ms)
                if len(self._detection_ms_samples) > self._config.fps_window:
                    self._detection_ms_samples.pop(0)

                # ---- Pose detection ----
                pose = None
                body_frame = None
                if self._pose_tracker is not None:
                    t1 = time.perf_counter()
                    try:
                        pose = self._pose_tracker.detect(frame.data)
                        if pose is not None:
                            body_frame = self._body_frame_extractor.extract(pose)
                    except Exception:
                        log.exception("preview.pose_error")
                    pose_ms = (time.perf_counter() - t1) * 1000
                    self._pose_ms_samples.append(pose_ms)
                    if len(self._pose_ms_samples) > self._config.fps_window:
                        self._pose_ms_samples.pop(0)

                self._last_pose = pose
                self._last_body_frame = body_frame

                # ---- Gesture recognition (dengan body frame) ----
                if self._gesture_engine is not None:
                    self._gesture_engine.set_body_frame(body_frame)
                    self._process_gestures(hands, frame.timestamp)

                self._last_hands = hands

                # ---- Render ----
                canvas = self._render(frame, hands, pose, body_frame)

                # ---- Recording ----
                if self._recording and self._video_writer is not None:
                    self._video_writer.write(canvas)

                # ---- Show ----
                cv2.imshow(self._config.window_name, canvas)

                # ---- Key handling ----
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, frame):
                    break

                self._frame_count += 1
                self._frame_timestamps.append(time.perf_counter())
                if len(self._frame_timestamps) > self._config.fps_window:
                    self._frame_timestamps.pop(0)

        finally:
            self._cleanup()

    def stop(self) -> None:
        """Request stop (dipanggil dari signal handler)."""
        cv2.waitKey(1)

    # ---------- Gesture processing ----------

    def _process_gestures(self, hands: list, timestamp: float) -> None:
        if not hands:
            return

        for hand in hands:
            try:
                smoothed_list = self._gesture_engine.process_hand(hand, timestamp)
            except Exception:
                log.exception("preview.gesture_error")
                continue

            for smoothed in smoothed_list:
                self._last_gestures[hand.handedness] = _HandGestureState(
                    gesture=smoothed.gesture,
                    confidence=smoothed.confidence,
                    timestamp=timestamp,
                )
                log.info(
                    "preview.gesture_confirmed",
                    gesture=smoothed.gesture.value,
                    handedness=hand.handedness,
                    confidence=round(smoothed.confidence, 2),
                    duration_ms=round(smoothed.duration_ms, 0),
                )

    # ---------- Rendering ----------

    def _render(
        self,
        frame: Frame,
        hands: list,
        pose: Any,
        body_frame: Any,
    ) -> np.ndarray:
        """Render frame + overlay."""
        canvas = frame.data.copy()

        # Mirror (selfie view)
        if self._config.mirror:
            canvas = cv2.flip(canvas, 1)
            hands = self._mirror_hands(hands)
            pose = self._mirror_pose(pose)
            body_frame = self._mirror_body_frame(body_frame)

        # ---- Draw pose skeleton (di belakang hand) ----
        if self._config.show_pose and pose is not None:
            draw_pose_skeleton(canvas, pose, line_thickness=2, dot_radius=3)

        # ---- Draw body frame axes ----
        if self._config.show_body_frame and body_frame is not None:
            draw_body_frame(canvas, body_frame, axis_length=0.12, thickness=2)

        # ---- Draw hand skeletons ----
        if self._config.show_skeleton:
            for hand in hands:
                draw_hand_skeleton(
                    canvas,
                    hand,
                    line_thickness=2,
                    dot_radius=4,
                    highlight_tips=True,
                    show_labels=self._config.show_landmark_names,
                )

        # ---- Hand labels ----
        if self._config.show_labels:
            for hand in hands:
                draw_hand_label(canvas, hand)

        # ---- Gesture overlay ----
        if self._config.show_gesture:
            self._draw_gestures(canvas, hands)

        # ---- Feature debug panel ----
        if self._config.show_features:
            for hand in hands:
                draw_feature_debug(canvas, hand, position=(20, 220))

        # ---- Crosshair ----
        if self._config.show_crosshair:
            draw_center_crosshair(canvas)

        # ---- HUD ----
        if self._config.show_hud:
            self._draw_hud(canvas, hands, pose, body_frame, frame)

        # ---- Resize ----
        canvas = self._resize_to_window(canvas)

        return canvas

    def _draw_gestures(self, canvas: np.ndarray, hands: list) -> None:
        """Draw gesture name di bawah hand label."""
        h, w = canvas.shape[:2]
        from nexus.vision.hand_tracking.landmarks import Landmark

        for hand in hands:
            state = self._last_gestures.get(hand.handedness)
            if state is None or state.gesture == Gesture.UNKNOWN:
                continue

            wrist = hand.landmarks[int(Landmark.WRIST)]
            px = int(wrist[0] * w)
            py = int(wrist[1] * h) + 20

            text = f"{state.gesture.value} ({state.confidence:.2f})"

            # Background
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                canvas,
                (px - 4, py - text_h - 6),
                (px + text_w + 4, py + baseline - 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                canvas,
                text,
                (px, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_hud(
        self,
        canvas: np.ndarray,
        hands: list,
        pose: Any,
        body_frame: Any,
        frame: Frame,
    ) -> None:
        fps = self._compute_fps()
        avg_detection_ms = self._compute_avg(self._detection_ms_samples)
        avg_pose_ms = self._compute_avg(self._pose_ms_samples)

        lines = [
            f"Frame:      {self._frame_count}",
            f"Resolution: {frame.width}x{frame.height}",
            f"Hands:      {len(hands)}",
            f"Detect:     {avg_detection_ms:.1f} ms",
        ]

        # Pose info
        if self._pose_tracker is not None:
            if pose is not None:
                vis = float(pose.visibility.mean())
                bf_status = "yes" if body_frame is not None else "no"
                lines.append(f"Pose:       yes (vis={vis:.2f})")
                lines.append(f"Body frame: {bf_status}")
                lines.append(f"Pose time:  {avg_pose_ms:.1f} ms")
            else:
                lines.append("Pose:       no")

        # Gesture summary
        gesture_lines = []
        for handedness, state in self._last_gestures.items():
            if state.gesture != Gesture.UNKNOWN:
                gesture_lines.append(
                    f"  {handedness}: {state.gesture.value} ({state.confidence:.2f})"
                )
        if gesture_lines:
            lines.append("Gestures:")
            lines.extend(gesture_lines)

        if self._recording:
            lines.append(f"● REC: {self._recording_path.name}")

        draw_hud_text(
            canvas,
            lines,
            position=(20, 40),
            font_scale=0.55,
            color=(0, 255, 0),
            line_height=22,
            background=True,
        )

        draw_fps_indicator(canvas, fps, font_scale=0.7)

        # Help text
        help_lines = [
            "q: quit | s: snapshot | r: record",
            "k: skeleton | l: labels | g: gesture | f: features",
            "p: pose | b: body frame | c: crosshair | m: mirror | h: hud",
        ]
        draw_hud_text(
            canvas,
            help_lines,
            position=(20, canvas.shape[0] - 55),
            font_scale=0.42,
            color=(180, 180, 180),
            line_height=17,
            background=True,
        )

    def _resize_to_window(self, canvas: np.ndarray) -> np.ndarray:
        """Resize canvas ke window size (maintain aspect ratio)."""
        cw, ch = self._config.window_width, self._config.window_height
        if cw <= 0 or ch <= 0:
            return canvas

        h, w = canvas.shape[:2]
        if (w, h) == (cw, ch):
            return canvas

        src_ar = w / h
        dst_ar = cw / ch

        if src_ar > dst_ar:
            new_w = cw
            new_h = int(cw / src_ar)
            resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pad_top = (ch - new_h) // 2
            pad_bottom = ch - new_h - pad_top
            return cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
        else:
            new_h = ch
            new_w = int(ch * src_ar)
            resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pad_left = (cw - new_w) // 2
            pad_right = cw - new_w - pad_left
            return cv2.copyMakeBorder(
                resized, 0, 0, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )

    # ---------- Key handling ----------

    def _handle_key(self, key: int, frame: Frame) -> bool:
        """Handle keyboard input. Returns False untuk quit."""
        if key == 0xFF:
            return True

        key_char = chr(key) if key < 128 else ""

        if key == 27 or key_char == "q":
            log.info("preview.quit_by_user")
            return False

        elif key_char == "s":
            self._take_snapshot(frame)

        elif key_char == "r":
            self._toggle_recording(frame)

        elif key_char == "l":
            self._config.show_labels = not self._config.show_labels
            log.info("preview.toggle", setting="labels", value=self._config.show_labels)

        elif key_char == "k":
            self._config.show_skeleton = not self._config.show_skeleton
            log.info("preview.toggle", setting="skeleton", value=self._config.show_skeleton)

        elif key_char == "c":
            self._config.show_crosshair = not self._config.show_crosshair
            log.info("preview.toggle", setting="crosshair", value=self._config.show_crosshair)

        elif key_char == "m":
            self._config.mirror = not self._config.mirror
            log.info("preview.toggle", setting="mirror", value=self._config.mirror)

        elif key_char == "h":
            self._config.show_hud = not self._config.show_hud
            log.info("preview.toggle", setting="hud", value=self._config.show_hud)

        elif key_char == "g":
            self._config.show_gesture = not self._config.show_gesture
            log.info("preview.toggle", setting="gesture", value=self._config.show_gesture)

        elif key_char == "f":
            self._config.show_features = not self._config.show_features
            log.info("preview.toggle", setting="features", value=self._config.show_features)

        elif key_char == "p":
            self._config.show_pose = not self._config.show_pose
            log.info("preview.toggle", setting="pose", value=self._config.show_pose)

        elif key_char == "b":
            self._config.show_body_frame = not self._config.show_body_frame
            log.info("preview.toggle", setting="body_frame", value=self._config.show_body_frame)

        return True

    # ---------- Snapshot ----------

    def _take_snapshot(self, frame: Frame) -> None:
        canvas = self._render(frame, self._last_hands, self._last_pose, self._last_body_frame)
        filename = f"snapshot_{int(time.time() * 1000)}.png"
        path = self._config.snapshot_dir / filename
        cv2.imwrite(str(path), canvas)
        log.info("preview.snapshot_saved", path=str(path))

    # ---------- Recording ----------

    def _toggle_recording(self, frame: Frame) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording(frame)

    def _start_recording(self, frame: Frame) -> None:
        filename = f"recording_{int(time.time() * 1000)}.mp4"
        path = self._config.recording_dir / filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            self._config.recording_fps,
            (frame.width, frame.height),
        )
        if not writer.isOpened():
            log.error("preview.recording_failed", path=str(path))
            return

        self._video_writer = writer
        self._recording_path = path
        self._recording = True
        log.info("preview.recording_started", path=str(path))

    def _stop_recording(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        log.info(
            "preview.recording_stopped",
            path=str(self._recording_path) if self._recording_path else None,
        )
        self._recording = False
        self._recording_path = None

    # ---------- Helpers ----------

    def _setup_window(self) -> None:
        cv2.namedWindow(self._config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self._config.window_name,
            self._config.window_width,
            self._config.window_height,
        )

    def _cleanup(self) -> None:
        if self._recording:
            self._stop_recording()
        cv2.destroyAllWindows()

        elapsed = time.perf_counter() - self._start_time if self._start_time else 0
        avg_fps = self._frame_count / elapsed if elapsed > 0 else 0

        log.info(
            "preview.stopped",
            frames=self._frame_count,
            elapsed_s=round(elapsed, 1),
            avg_fps=round(avg_fps, 2),
        )

    def _compute_fps(self) -> float:
        if len(self._frame_timestamps) < 2:
            return 0.0
        elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_timestamps) - 1) / elapsed

    @staticmethod
    def _compute_avg(samples: list[float]) -> float:
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def _mirror_hands(self, hands: list) -> list:
        """Mirror hand landmarks secara horizontal (selfie view)."""
        mirrored = []
        for hand in hands:
            mirrored_lms = hand.landmarks.copy()
            mirrored_lms[:, 0] = 1.0 - mirrored_lms[:, 0]

            if hand.world_landmarks is not None:
                mw = hand.world_landmarks.copy()
                mw[:, 0] = -mw[:, 0]
            else:
                mw = None

            # Swap handedness
            new_handedness = "Left" if hand.handedness == "Right" else "Right"

            from nexus.vision.hand_tracking.landmarks import HandLandmarks
            mirrored.append(HandLandmarks(
                landmarks=mirrored_lms,
                handedness=new_handedness,
                confidence=hand.confidence,
                world_landmarks=mw,
            ))
        return mirrored

    def _mirror_pose(self, pose: Any) -> Any:
        """Mirror pose landmarks."""
        if pose is None:
            return None
        mirrored_lms = pose.landmarks.copy()
        mirrored_lms[:, 0] = 1.0 - mirrored_lms[:, 0]

        if pose.world_landmarks is not None:
            mw = pose.world_landmarks.copy()
            mw[:, 0] = -mw[:, 0]
        else:
            mw = None

        from nexus.vision.pose.landmarks import PoseLandmarks
        return PoseLandmarks(
            landmarks=mirrored_lms,
            visibility=pose.visibility.copy(),
            world_landmarks=mw,
        )

    def _mirror_body_frame(self, body_frame: Any) -> Any:
        """Mirror body frame axes."""
        if body_frame is None:
            return None

        from nexus.vision.context.body_frame import BodyFrame

        # Mirror x-component dari semua vector
        new_origin = body_frame.origin.copy()
        new_origin[0] = 1.0 - new_origin[0]

        new_forward = body_frame.forward.copy()
        new_forward[0] = -new_forward[0]

        new_up = body_frame.up.copy()
        new_up[0] = -new_up[0]

        new_right = body_frame.right.copy()
        new_right[0] = -new_right[0]

        return BodyFrame(
            origin=new_origin,
            forward=new_forward,
            up=new_up,
            right=new_right,
            shoulder_width=body_frame.shoulder_width,
            visibility=body_frame.visibility,
        )